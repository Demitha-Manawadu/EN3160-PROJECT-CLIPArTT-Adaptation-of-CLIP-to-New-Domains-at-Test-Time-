import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torchvision import datasets
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from clip import load, tokenize

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load("ViT-B/32", device=device)

# Load CIFAR-10 test set
cifar = datasets.CIFAR10(root=os.path.expanduser("~/.torch/data"), train=False, download=True)

# Predefined classes
class_names = cifar.classes
texts = tokenize([f"a photo of a {c}" for c in class_names]).to(device)
text_features = model.encode_text(texts)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Show N random images with predictions
n = 8
indices = torch.randint(0, len(cifar), (n,))
plt.figure(figsize=(12, 6))

for i, idx in enumerate(indices):
    image, label = cifar[idx]
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_prob, top_label = probs[0].topk(1)

    plt.subplot(2, 4, i + 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title(
        f"Pred: {class_names[top_label]} ({top_prob.item()*100:.1f}%)\nTrue: {class_names[label]}",
        fontsize=9,
    )

plt.tight_layout()
plt.show()
