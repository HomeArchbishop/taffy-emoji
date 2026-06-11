import json
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from .config import CACHE_DIR
from .model import IMG_SIZE, MEAN, STD

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {".webp", ".awebp", ".gif", ".jpeg", ".jpg", ".png"}

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

infer_tf = val_tf


def open_image(path: Path) -> Image.Image:
    img = Image.open(path)
    if hasattr(img, "n_frames") and img.n_frames > 1:
        img.seek(0)
    return img.convert("RGB")


def list_images(image_dir: Path) -> list[str]:
    return sorted(f.name for f in image_dir.iterdir()
                  if f.suffix.lower() in IMAGE_EXTS)


def _cache_complete(cache_dir: Path, n: int) -> bool:
    labels_file = cache_dir / "labels.json"
    images_dir  = cache_dir / "images"
    if not labels_file.exists() or not images_dir.is_dir():
        return False
    labels = json.loads(labels_file.read_text())["labels"]
    if len(labels) != n:
        return False
    return sum(1 for _ in images_dir.glob("*.jpg")) == n


def ensure_cached(hf_ds, cache_dir: Path = CACHE_DIR) -> list[int]:
    """Export HF images to local JPEG cache for fast DataLoader reads."""
    images_dir  = cache_dir / "images"
    labels_file = cache_dir / "labels.json"
    images_dir.mkdir(parents=True, exist_ok=True)

    n = len(hf_ds)
    if _cache_complete(cache_dir, n):
        return json.loads(labels_file.read_text())["labels"]

    labels = [int(x) for x in hf_ds["label"]]
    missing = [i for i in range(n) if not (images_dir / f"{i}.jpg").exists()]
    if missing:
        print(f"caching {len(missing)} images → {cache_dir}")
        for i in tqdm(missing, desc="cache"):
            hf_ds[i]["image"].convert("RGB").save(images_dir / f"{i}.jpg", "JPEG", quality=95)

    labels_file.write_text(json.dumps({"labels": labels}))
    return labels


class CachedEmojiDataset(Dataset):
    """Reads pre-cached local JPEGs."""

    def __init__(self, cache_dir: Path, indices: list[int], labels: list[int], transform):
        self.images_dir = cache_dir / "images"
        self.indices    = indices
        self.labels     = labels
        self.transform  = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return self.transform(open_image(self.images_dir / f"{i}.jpg")), torch.tensor(
            self.labels[i], dtype=torch.float32
        )


class HFEmojiDataset(Dataset):
    """Wraps a HuggingFace dataset for training/validation."""

    def __init__(self, hf_ds, indices: list[int], transform):
        self.hf_ds     = hf_ds
        self.indices   = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.hf_ds[self.indices[idx]]
        return self.transform(item["image"].convert("RGB")), torch.tensor(item["label"], dtype=torch.float32)


class InferDataset(Dataset):
    """Unlabeled dataset for inference."""

    def __init__(self, filenames: list[str], image_dir: Path):
        self.filenames = filenames
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        try:
            img = open_image(self.image_dir / fname)
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return infer_tf(img), fname
