import os
import sys
import time
import argparse
import tarfile
import urllib.request
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

# Make "clip" importable when running from repo root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from clip import load, tokenize

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

CIFAR10C_URL = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"
CIFAR10C_DIR = "CIFAR-10-C"

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression"
]

TEMPLATE = "a photo of a {}"
CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# ---------------------------------------------------------------------
# Download / extract
# ---------------------------------------------------------------------

def _download_with_retries(url: str, dst: str, max_retries: int = 5) -> None:
    backoff = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            urllib.request.urlretrieve(url, dst)
            return
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(backoff)
            backoff *= 2

def ensure_cifar10c(root: str) -> None:
    os.makedirs(root, exist_ok=True)
    arc_tar = os.path.join(root, "CIFAR-10-C.tar")
    target_dir = os.path.join(root, CIFAR10C_DIR)

    # If already extracted with labels present, done
    if os.path.isdir(target_dir) and os.path.isfile(os.path.join(target_dir, "labels.npy")):
        return

    # If the archive is not there, download with retries
    if not os.path.isfile(arc_tar):
        print(f"Downloading CIFAR-10-C to {arc_tar}")
        _download_with_retries(CIFAR10C_URL, arc_tar)
        print("Download complete")

    # Extract
    print(f"Extracting {arc_tar}")
    with tarfile.open(arc_tar, "r") as tf:
        tf.extractall(path=root)
    print("Extraction complete")

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class CIFAR10C(Dataset):
    """
    CIFAR-10-C loader.

    severity: 1..5
    corruption: one of CORRUPTIONS or "all"

    Loads labels once, memory-maps the selected corruption arrays.
    """
    def __init__(self, root: str, severity: int = 1, corruption: str = "all", transform=None):
        assert 1 <= severity <= 5, "severity must be in [1..5]"
        self.root = root
        self.severity = severity
        self.transform = transform

        base = os.path.join(root, CIFAR10C_DIR)
        labels_path = os.path.join(base, "labels.npy")
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(f"labels.npy not found at {labels_path}. Run ensure_cifar10c(root) first.")

        # Labels are the same across corruptions, length 10000
        self.labels_block = np.load(labels_path)

        # Select corruption set
        if corruption == "all":
            self.corruptions: List[str] = CORRUPTIONS
        else:
            if corruption not in CORRUPTIONS:
                raise ValueError(f"Unknown corruption: {corruption}")
            self.corruptions = [corruption]

        # Build index across chosen corruptions
        self.index: List[Tuple[str, int]] = []
        for cname in self.corruptions:
            for i in range(10000):
                self.index.append((cname, i))

        # Memory map arrays once per corruption to avoid reloading each sample
        self.arrs: Dict[str, np.memmap] = {}
        for cname in self.corruptions:
            path = os.path.join(base, f"{cname}.npy")
            # Use memmap to avoid loading full 50k x 32 x 32 x 3 for all corruptions
            self.arrs[cname] = np.load(path, mmap_mode="r")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        cname, i = self.index[idx]
        arr = self.arrs[cname]               # shape (50000, 32, 32, 3)
        off = (self.severity - 1) * 10000
        img_np = np.array(arr[off + i])      # copy out the small 32x32 slice
        label = int(self.labels_block[i])

        img = Image.fromarray(img_np)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate(model_name="ViT-B/32", batch_size=128, workers=4, severity=1, corruption="all"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, preprocess = load(model_name, device=device)

    root = os.path.expanduser("~/.torch/data")
    ensure_cifar10c(root)

    ds = CIFAR10C(root=root, severity=severity, corruption=corruption, transform=preprocess)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    prompts = [TEMPLATE.format(c) for c in CIFAR10_CLASSES]
    text = tokenize(prompts).to(device)
    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)  # [10, D]

    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        logits = image_features @ text_features.T
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100.0 * correct / total
    print(f"CIFAR-10-C zero-shot | model={model_name} | corruption={corruption} | severity={severity} | acc={acc:.2f}%")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("Zero-shot CLIP on CIFAR-10-C")
    ap.add_argument("--model", type=str, default="ViT-B/32")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--severity", type=int, default=1, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--corruption", type=str, default="all",
                    help="one of: " + ",".join(CORRUPTIONS) + " or 'all'")
    args = ap.parse_args()

    evaluate(model_name=args.model,
             batch_size=args.batch_size,
             workers=args.workers,
             severity=args.severity,
             corruption=args.corruption)

if __name__ == "__main__":
    main()
