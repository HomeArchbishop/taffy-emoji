#!/usr/bin/env python3
"""Run inference on images.
Usage:
  uv run python -m src.predict --dir data/test       # dir → outputs/<dir>_predictions.json
  uv run python -m src.predict path/to/image.jpg     # single image → stdout
  uv run python -m src.predict --threshold 0.6
"""

import argparse
import json
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CKPT_DIR, OUTPUTS_DIR, device
from .dataset import InferDataset, list_images
from .model import build_model

warnings.filterwarnings("ignore")

CKPT_FILE = CKPT_DIR / "best.pth"


def load_model():
    ckpt  = torch.load(CKPT_FILE, map_location=device, weights_only=True)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    print(f"checkpoint  epoch={ckpt['epoch']}  val_f1={ckpt['f1']:.4f}  val_auc={ckpt['auc']:.4f}")
    return model


@torch.no_grad()
def run(model, filenames: list[str], threshold: float, image_dir: Path) -> dict:
    ds     = InferDataset(filenames, image_dir=image_dir)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    results = {}
    for imgs, fnames in tqdm(loader, desc="predict"):
        probs = model(imgs.to(device)).squeeze(1).sigmoid().cpu().tolist()
        for fname, prob in zip(fnames, probs):
            results[fname] = {"prob": round(prob, 4), "label": int(prob >= threshold)}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image",      nargs="?",   type=Path, default=None, help="single image file")
    parser.add_argument("--dir",      type=Path,   default=None,            help="directory of images")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.image:
        if not args.image.is_absolute():
            args.image = Path.cwd() / args.image
        image_dir = args.image.parent
        filenames = [args.image.name]
    elif args.dir:
        image_dir = args.dir if args.dir.is_absolute() else Path.cwd() / args.dir
        filenames = list_images(image_dir)
        print(f"images: {len(filenames)}  dir: {image_dir}")
    else:
        parser.error("provide an image path or --dir")

    model   = load_model()
    results = run(model, filenames, args.threshold, image_dir)

    if args.image:
        fname = filenames[0]
        r = results[fname]
        print(f"{fname}  →  label={r['label']}  prob={r['prob']:.4f}")
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
