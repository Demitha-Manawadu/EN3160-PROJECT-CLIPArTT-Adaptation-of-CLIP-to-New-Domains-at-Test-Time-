import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

# Add repo root so "clip" is importable
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from clip import load, tokenize

TEMPLATE = "a photo of a {}"
CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

@torch.no_grad()
def main(model_name="ViT-B/32", batch_size=128, workers=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, preprocess = load(model_name, device=device)

    root = os.path.expanduser("~/.torch/data")
    ds = datasets.CIFAR10(root=root, train=False, download=True, transform=preprocess)

    prompts = [TEMPLATE.format(c) for c in CIFAR10_CLASSES]
    text = tokenize(prompts).to(device)
    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)  # [10, D]

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)  # [B, D]
        logits = image_features @ text_features.T  # [B, 10]
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100.0 * correct / total
    print(f"Zero-shot accuracy on CIFAR-10 test: {acc:.2f}% using {model_name}")

if __name__ == "__main__":
    main()
