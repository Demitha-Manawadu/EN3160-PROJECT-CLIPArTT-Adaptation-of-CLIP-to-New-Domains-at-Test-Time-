import os
import sys
from torchvision import datasets

def export(split="test", out_dir="cifar10_test_export"):
    os.makedirs(out_dir, exist_ok=True)
    is_train = (split == "train")
    ds = datasets.CIFAR10(root=os.path.expanduser("~/.torch/data"),
                          train=is_train, download=True)
    classes = ds.classes
    for c in classes:
        os.makedirs(os.path.join(out_dir, c), exist_ok=True)

    # torchvision CIFAR returns PIL images
    for idx, (img, label) in enumerate(ds):
        c = classes[label]
        fname = os.path.join(out_dir, c, f"{idx:05d}.png")
        img.save(fname)

    print(f"Exported {len(ds)} images to {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    out = sys.argv[2] if len(sys.argv) > 2 else f"cifar10_{split}_export"
    export(split, out)
