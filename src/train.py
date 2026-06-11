#!/usr/bin/env python3
"""Train binary classifier.
Usage:
  uv run python -m src.train
  uv run python -m src.train --resume
"""

import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .config import CKPT_DIR, device
from .dataset import HFEmojiDataset, train_tf, val_tf
from .model import build_model

warnings.filterwarnings("ignore", category=UserWarning)

EPOCHS = 50
BATCH  = 32
LR     = 1e-4
WD     = 1e-2
SEED   = 42


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, preds, targets = 0.0, [], []
    for imgs, labels in tqdm(loader, leave=False, desc="train"):
        imgs, labels = imgs.to(device), labels.to(device)
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
        imgs, labels = imgs.to(device), labels.to(device)
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
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"device: {device}")

    hf_ds = load_dataset("homearchbishop/ace-taffy-images", split="train")
    lbls  = hf_ds["label"]
    n0, n1 = lbls.count(0), lbls.count(1)
    print(f"samples: {len(lbls)}  (0={n0}  1={n1})")

    indices = list(range(len(hf_ds)))
    tr_idx, va_idx, tr_l, _ = train_test_split(
        indices, lbls, test_size=0.2, stratify=lbls, random_state=SEED
    )

    tr_ds = HFEmojiDataset(hf_ds, tr_idx, train_tf)
    va_ds = HFEmojiDataset(hf_ds, va_idx, val_tf)

    counts  = np.bincount(tr_l)
    sampler = WeightedRandomSampler([1.0 / counts[label] for label in tr_l], len(tr_l))
    tr_loader = DataLoader(tr_ds, batch_size=BATCH, sampler=sampler, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=BATCH, shuffle=False,  num_workers=0)

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
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_f1 = 0.0
    hdr = (f"{'ep':>3} | {'tr_loss':>7} {'tr_acc':>6} {'tr_f1':>5} | "
           f"{'va_loss':>7} {'va_acc':>6} {'P':>5} {'R':>5} {'F1':>5} {'AUC':>5}")
    print(f"\n{hdr}\n{'-' * len(hdr)}")

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, tr_loader, criterion, optimizer)
        va_loss, va_acc, p, r, f1, auc = val_epoch(model, va_loader, criterion)
        scheduler.step()

        marker = " ✓" if f1 > best_f1 else ""
        print(f"{ep:>3} | {tr_loss:7.4f} {tr_acc:6.3f} {tr_f1:5.3f} | "
              f"{va_loss:7.4f} {va_acc:6.3f} {p:5.3f} {r:5.3f} {f1:5.3f} {auc:5.3f}{marker}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"epoch": ep, "state": model.state_dict(),
                        "f1": f1, "acc": va_acc, "auc": auc},
                       CKPT_DIR / "best.pth")

    print(f"\nbest val F1={best_f1:.4f}  saved → {CKPT_DIR}/best.pth")


if __name__ == "__main__":
    main()
