#!/usr/bin/env python3
"""Run inference on images.
Usage:
  uv run python -m src.predict --dir data/test       # dir → outputs/<dir>_predictions.json
  uv run python -m src.predict path/to/image.jpg     # single image → stdout
  uv run python -m src.predict https://example.com/x.jpg
  uv run python -m src.predict --threshold 0.6
  uv run python -m src.predict --val                 # threshold sweep on val split
"""

import argparse
import json
import warnings
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from datasets import load_dataset
from PIL import Image

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CACHE_DIR, CKPT_DIR, OUTPUTS_DIR, device
from .dataset import CachedEmojiDataset, InferDataset, ensure_cached, infer_tf, list_images, val_tf
from .model import build_model

warnings.filterwarnings("ignore")

CKPT_FILE = CKPT_DIR / "best.pth"
SEED = 42


def _is_url(s: str) -> bool:
    return urlparse(s).scheme in ("http", "https")


def _load_image_url(url: str) -> Image.Image:
    req = Request(url, headers={"User-Agent": "taffy-emoji"})
    with urlopen(req, timeout=30) as resp:
        img = Image.open(BytesIO(resp.read()))
    if hasattr(img, "n_frames") and img.n_frames > 1:
        img.seek(0)
    return img.convert("RGB")


@torch.no_grad()
def predict_one(model, img: Image.Image, threshold: float) -> tuple[float, int]:
    x = infer_tf(img).unsqueeze(0).to(device)
    prob = model(x).squeeze().sigmoid().item()
    return prob, int(prob >= threshold)


def load_model(ckpt_path: Path = CKPT_FILE):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    print(f"checkpoint  epoch={ckpt['epoch']}  val_f1={ckpt['f1']:.4f}  val_auc={ckpt['auc']:.4f}")
    return model


@torch.no_grad()
def collect_val_probs(model) -> tuple[list[float], list[int]]:
    hf_ds = load_dataset("homearchbishop/ace-taffy-images", split="train")
    lbls = ensure_cached(hf_ds, CACHE_DIR)
    indices = list(range(len(lbls)))
    _, va_idx, _, _ = train_test_split(
        indices, lbls, test_size=0.2, stratify=lbls, random_state=SEED
    )
    va_ds = CachedEmojiDataset(CACHE_DIR, va_idx, lbls, val_tf)
    loader = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=0)

    probs, targets = [], []
    for imgs, labels in tqdm(loader, desc="val"):
        prob = model(imgs.to(device)).squeeze(1).sigmoid()
        probs.extend(prob.cpu().tolist())
        targets.extend(labels.int().tolist())
    return probs, targets


def sweep_thresholds(probs: list[float], targets: list[int], highlight: float | None = None):
    auc = roc_auc_score(targets, probs) if len(set(targets)) > 1 else 0.0
    print(f"val samples: {len(targets)}  auc={auc:.4f}")

    hdr = f"{'thr':>5} | {'acc':>5} {'P':>5} {'R':>5} {'F1':>5}"
    print(hdr)
    print("-" * len(hdr))

    best_f1, best_thr = 0.0, 0.5
    for i in range(2, 19):
        thr = round(i * 0.05, 2)
        preds = [1 if p >= thr else 0 for p in probs]
        p, r, f1, _ = precision_recall_fscore_support(
            targets, preds, average="binary", zero_division=0
        )
        acc = accuracy_score(targets, preds)
        marker = ""
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
            marker = " *"
        if highlight is not None and thr == highlight:
            marker += " ←"
        print(f"{thr:5.2f} | {acc:5.3f} {p:5.3f} {r:5.3f} {f1:5.3f}{marker}")

    print(f"\nbest F1={best_f1:.4f} at threshold={best_thr:.2f}")


def run_val(model, highlight: float | None = None):
    probs, targets = collect_val_probs(model)
    sweep_thresholds(probs, targets, highlight=highlight)


@torch.no_grad()
def run(model, filenames: list[str], threshold: float, image_dir: Path) -> dict:
    ds = InferDataset(filenames, image_dir=image_dir)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    results = {}
    for imgs, fnames in tqdm(loader, desc="predict"):
        probs = model(imgs.to(device)).squeeze(1).sigmoid().cpu().tolist()
        for fname, prob in zip(fnames, probs):
            results[fname] = {"prob": round(prob, 4), "label": int(prob >= threshold)}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default=None, help="local path or http(s) URL")
    parser.add_argument("--dir", type=Path, default=None, help="directory of images")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--val", action="store_true",
                        help="run on validation split and print per-threshold metrics")
    parser.add_argument("--ckpt", type=Path, default=CKPT_FILE, help="checkpoint path")
    args = parser.parse_args()

    if args.val:
        run_val(load_model(args.ckpt), highlight=args.threshold)
        return

    if args.image:
        if _is_url(args.image):
            model = load_model(args.ckpt)
            prob, label = predict_one(model, _load_image_url(args.image), args.threshold)
            print(f"{args.image}  →  label={label}  prob={prob:.4f}")
            return
        image_path = Path(args.image)
        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path
        image_dir = image_path.parent
        filenames = [image_path.name]
    elif args.dir:
        image_dir = args.dir if args.dir.is_absolute() else Path.cwd() / args.dir
        filenames = list_images(image_dir)
        print(f"images: {len(filenames)}  dir: {image_dir}")
    else:
        parser.error("provide an image path, --dir, or --val")

    model = load_model(args.ckpt)
    results = run(model, filenames, args.threshold, image_dir)

    if args.image:
        r = results[filenames[0]]
        print(f"{filenames[0]}  →  label={r['label']}  prob={r['prob']:.4f}")
    else:
        out_file = OUTPUTS_DIR / f"{image_dir.name}_predictions.json"
        existing = json.loads(out_file.read_text()) if out_file.exists() else {}
        existing.update(results)
        out_file.write_text(json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True))
        n1 = sum(1 for v in results.values() if v["label"] == 1)
        print(f"results: 0={len(results)-n1}  1={n1}  threshold={args.threshold}")
        print(f"saved → {out_file}")


if __name__ == "__main__":
    main()
