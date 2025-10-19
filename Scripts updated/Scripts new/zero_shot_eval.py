import os
import sys
import torch
from torchvision import datasets
from torch.utils.data import DataLoader

# Add repo root for "clip"
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from clip import load, tokenize

TEMPLATE = "a photo of a {}"

@torch.no_grad()
def main(root, model_name="ViT-B/32", batch_size=64, workers=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, preprocess = load(model_name, device=device)

    ds = datasets.ImageFolder(root=root, transform=preprocess)
    classnames = ds.classes
    prompts = [TEMPLATE.format(c.replace("_", " ").strip()) for c in classnames]
    text = tokenize(prompts).to(device)

    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)  # [C, D]

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        logits = image_features @ text_features.T
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100.0 * correct / total
    print(f"Zero-shot accuracy on {root}: {acc:.2f}% using {model_name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/zero_shot_eval.py <imagefolder_root>")
        sys.exit(1)
    main(sys.argv[1])
