#!/usr/bin/env python3
"""Train binary classifier.
Usage:
  uv run python -m src.train
  uv run python -m src.train --resume
  uv run python -m src.train --epochs 30 --patience 5
"""

import argparse
import os
import warnings

from datasets import load_dataset

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .config import CACHE_DIR, CKPT_DIR, device
from .dataset import CachedEmojiDataset, ensure_cached, train_tf, val_tf
from .model import build_model

warnings.filterwarnings("ignore", category=UserWarning)

EPOCHS   = 25
PATIENCE = 8
BATCH    = 64
LR       = 1e-4
WD       = 1e-2
SEED     = 42

_use_cuda = device.type == "cuda"


def _to_device(t):
    return t.to(device, non_blocking=_use_cuda)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, preds, targets = 0.0, [], []
    for imgs, labels in tqdm(loader, leave=False, desc="train"):
        imgs, labels = _to_device(imgs), _to_device(labels)
        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        preds.extend((logits.detach().sigmoid() > 0.5).cpu().int().tolist())
        targets.extend(labels.cpu().int().tolist())
    _, _, f1, _ = precision_recall_fscore_support(targets, preds, average="binary", zero_division=0)
    return total_loss / len(loader.dataset), accuracy_score(targets, preds), f1


@torch.no_grad()
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, preds, probs, targets = 0.0, [], [], []
    for imgs, labels in loader:
        imgs, labels = _to_device(imgs), _to_device(labels)
        logits = model(imgs).squeeze(1)
        total_loss += criterion(logits, labels).item() * len(labels)
        prob = logits.sigmoid()
        probs.extend(prob.cpu().tolist())
        preds.extend((prob > 0.5).cpu().int().tolist())
        targets.extend(labels.cpu().int().tolist())
    p, r, f1, _ = precision_recall_fscore_support(targets, preds, average="binary", zero_division=0)
    auc = roc_auc_score(targets, probs) if len(set(targets)) > 1 else 0.0
    return total_loss / len(loader.dataset), accuracy_score(targets, preds), p, r, f1, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="load checkpoints/best.pth weights as starting point")
    parser.add_argument("--batch-size", type=int, default=BATCH, help=f"batch size (default {BATCH})")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"max epochs (default {EPOCHS})")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help=f"early stop after N epochs without val F1 gain (default {PATIENCE})")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"device: {device}")

    hf_ds = load_dataset("homearchbishop/ace-taffy-images", split="train")
    lbls  = ensure_cached(hf_ds, CACHE_DIR)
    n0, n1 = lbls.count(0), lbls.count(1)
    print(f"samples: {len(lbls)}  (0={n0}  1={n1})")

    indices = list(range(len(lbls)))
    tr_idx, va_idx, tr_l, _ = train_test_split(
        indices, lbls, test_size=0.2, stratify=lbls, random_state=SEED
    )

    tr_ds = CachedEmojiDataset(CACHE_DIR, tr_idx, lbls, train_tf)
    va_ds = CachedEmojiDataset(CACHE_DIR, va_idx, lbls, val_tf)

    counts  = np.bincount(tr_l)
    sampler = WeightedRandomSampler([1.0 / counts[label] for label in tr_l], len(tr_l))
    nw = min(4, os.cpu_count() or 1)
    dl_kw = {"batch_size": args.batch_size, "num_workers": nw, "pin_memory": _use_cuda}
    if nw > 0:
        dl_kw["persistent_workers"] = True
        dl_kw["prefetch_factor"] = 2
    tr_loader = DataLoader(tr_ds, sampler=sampler, **dl_kw)
    va_loader = DataLoader(va_ds, shuffle=False, **dl_kw)

    ckpt_path = CKPT_DIR / "best.pth"
    if args.resume and ckpt_path.exists():
        model = build_model(pretrained=False).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state"])
        print(f"resumed  epoch={ckpt['epoch']}  f1={ckpt['f1']:.4f}  auc={ckpt['auc']:.4f}")
    elif args.resume:
        print("no checkpoint found, starting from ImageNet weights")
        model = build_model(pretrained=True).to(device)
    else:
        model = build_model(pretrained=True).to(device)
    pos_weight = torch.tensor([counts[0] / counts[1]], dtype=torch.float32, device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_f1, best_ep, stall = 0.0, 0, 0
    hdr = (f"{'ep':>3} | {'tr_loss':>7} {'tr_acc':>6} {'tr_f1':>5} | "
           f"{'va_loss':>7} {'va_acc':>6} {'P':>5} {'R':>5} {'F1':>5} {'AUC':>5}")
    print(f"\n{hdr}\n{'-' * len(hdr)}")

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, tr_loader, criterion, optimizer)
        va_loss, va_acc, p, r, f1, auc = val_epoch(model, va_loader, criterion)
        scheduler.step()

        improved = f1 > best_f1
        marker = " ✓" if improved else ""
        print(f"{ep:>3} | {tr_loss:7.4f} {tr_acc:6.3f} {tr_f1:5.3f} | "
              f"{va_loss:7.4f} {va_acc:6.3f} {p:5.3f} {r:5.3f} {f1:5.3f} {auc:5.3f}{marker}")

        if improved:
            best_f1, best_ep, stall = f1, ep, 0
            torch.save({"epoch": ep, "state": model.state_dict(),
                        "f1": f1, "acc": va_acc, "auc": auc},
                       CKPT_DIR / "best.pth")
        else:
            stall += 1
            if stall >= args.patience:
                print(f"\nearly stop at epoch {ep}  (val F1 stalled {args.patience} epochs, best={best_f1:.4f} ep={best_ep})")
                break

    print(f"\nbest val F1={best_f1:.4f} at epoch {best_ep}  saved → {CKPT_DIR}/best.pth")


if __name__ == "__main__":
    main()
