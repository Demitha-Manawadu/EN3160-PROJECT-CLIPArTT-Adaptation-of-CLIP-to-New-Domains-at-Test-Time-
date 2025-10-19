import os
import sys
import argparse
from typing import List
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

# Make "clip" importable
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from clip import load, tokenize

TEMPLATE = "a photo of a {}"
CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
CIFAR10C_CORRUPTIONS = [
    "gaussian_noise","shot_noise","impulse_noise","defocus_blur",
    "glass_blur","motion_blur","zoom_blur","snow","frost","fog",
    "brightness","contrast","elastic_transform","pixelate","jpeg_compression"
]

# ---------------------- CIFAR-10-C Loader ----------------------
def find_cifar10c_root(root_hint: str) -> str:
    base = os.path.expanduser(root_hint)
    for name in ["CIFAR-10-C", "cifar10-c"]:
        p = os.path.join(base, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "labels.npy")):
            return p
    raise FileNotFoundError(f"Could not find CIFAR-10-C under {base}")

class CIFAR10C(Dataset):
    def __init__(self, root: str, corruption: str, severity: int, transform=None):
        assert corruption in CIFAR10C_CORRUPTIONS
        assert 1 <= severity <= 5
        base = find_cifar10c_root(root)
        data_path = os.path.join(base, f"{corruption}.npy")
        labels_path = os.path.join(base, "labels.npy")
        imgs = np.load(data_path)
        labels = np.load(labels_path)
        start, end = (severity-1)*10000, severity*10000
        self.imgs = imgs[start:end]
        self.labels = labels[start:end]
        self.transform = transform

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        img = Image.fromarray(self.imgs[i])
        if self.transform: img = self.transform(img)
        return img, int(self.labels[i])

# ---------------------- CLIP + Adaptation ----------------------
def set_visual_ln_trainable(model: nn.Module):
    for p in model.parameters(): p.requires_grad_(False)
    params = []
    for _, m in model.visual.named_modules():
        if isinstance(m, nn.LayerNorm) and getattr(m, "elementwise_affine", True):
            if getattr(m, "weight", None) is not None:
                m.weight.requires_grad_(True); params.append(m.weight)
            if getattr(m, "bias", None) is not None:
                m.bias.requires_grad_(True); params.append(m.bias)
    return params

@torch.no_grad()
def build_base_text_features(model, device):
    prompts = [TEMPLATE.format(c) for c in CIFAR10_CLASSES]
    text = tokenize(prompts).to(device)
    zt = model.encode_text(text)
    return zt / zt.norm(dim=1, keepdim=True)

def column_softmax(x, tau): return torch.softmax((x / tau).T, dim=1).T

def build_instance_prompts_from_topk(logits, classnames, k):
    topk = torch.topk(logits, k=k, dim=1).indices
    prompts = []
    for i in range(topk.size(0)):
        names = [classnames[idx] for idx in topk[i].tolist()]
        prompts.append(f"a photo of a {' or '.join(names)}")
    return prompts

def adapt_one_batch(model, images, base_text_features, tau, k, iters, lr, alpha, no_diag, conf_thresh):
    device = images.device
    eps = 1e-6
    params = set_visual_ln_trainable(model)
    optim_ = optim.Adam(params, lr=lr)
    for _ in range(iters):
        z_img = model.encode_image(images)
        z_img = z_img / (z_img.norm(dim=1, keepdim=True) + eps)
        logits_base = z_img @ base_text_features.T
        inst_prompts = build_instance_prompts_from_topk(logits_base, CIFAR10_CLASSES, k)
        tokens = tokenize(inst_prompts).to(device)
        zt_hat = model.encode_text(tokens)
        zt_hat = zt_hat / (zt_hat.norm(dim=1, keepdim=True) + eps)
        S_v = z_img @ z_img.T
        S_t = zt_hat @ zt_hat.T
        S = alpha * S_v + (1-alpha) * S_t
        if no_diag: S = S - torch.diag_embed(torch.diag(S))
        Q = column_softmax(S, tau=tau)
        inter_logits = (z_img @ zt_hat.T) / max(tau, eps)
        log_probs = torch.log_softmax(inter_logits, dim=1)
        if conf_thresh > 0:
            with torch.no_grad():
                probs = torch.softmax(inter_logits, dim=1)
                col_conf = probs.max(dim=0).values
                col_mask = (col_conf >= conf_thresh).float()
                col_w = col_mask / (col_mask.sum() + eps)
            loss = -((Q * log_probs).sum(dim=0) * col_w).sum()
        else:
            loss = -(Q * log_probs).sum(dim=0).mean()
        optim_.zero_grad(); loss.backward(); optim_.step()

# ---------------------- Visualization ----------------------
def show_bar(zs_acc, tta_acc, corruption, severity):
    plt.figure(figsize=(4,3))
    plt.bar(["Zero-shot","CLIPArTT"], [zs_acc, tta_acc], color=["gray","skyblue"])
    plt.ylabel("Accuracy (%)")
    plt.title(f"{corruption} | severity {severity}")
    for i, v in enumerate([zs_acc, tta_acc]):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center")
    plt.tight_layout()
    plt.show()

def show_grid(images, zs_logits, tta_logits, labels, classnames):
    import math
    n = min(16, images.size(0))
    rows = math.ceil(n/4)
    mean = torch.tensor([0.48145466,0.4578275,0.40821073],device=images.device).view(1,3,1,1)
    std = torch.tensor([0.26862954,0.26130258,0.27577711],device=images.device).view(1,3,1,1)
    imgs_disp = (images*std+mean).clamp(0,1).cpu()
    zs_pred = zs_logits.argmax(dim=1)
    tta_pred = tta_logits.argmax(dim=1)
    plt.figure(figsize=(10,7))
    for i in range(n):
        ax = plt.subplot(rows,4,i+1)
        ax.imshow(imgs_disp[i].permute(1,2,0))
        title = f"ZS:{classnames[zs_pred[i]]}\nTTA:{classnames[tta_pred[i]]}"
        color = "green" if tta_pred[i]==labels[i] else "red"
        ax.set_title(title,color=color,fontsize=8)
        ax.axis("off")
    plt.tight_layout(); plt.show()

# ---------------------- Main ----------------------
def main():
    p = argparse.ArgumentParser("Visual CLIPArTT Advanced")
    p.add_argument("--cifar-c-root", type=str, default="~/.torch/data")
    p.add_argument("--corruption", type=str, default="glass_blur")
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--conf-thresh", type=float, default=0.3)
    p.add_argument("--no-diag", action="store_true", default=True)
    p.add_argument("--model", type=str, default="ViT-B/32")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, preprocess = load(args.model, device=device)
    ds = CIFAR10C(root=args.cifar_c_root, corruption=args.corruption, severity=args.severity, transform=preprocess)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    base_text = build_base_text_features(model, device)

    with torch.no_grad():
        z_img = model.encode_image(images); z_img = z_img / z_img.norm(dim=1, keepdim=True)
        zs_logits = z_img @ base_text.T
    zs_acc = (zs_logits.argmax(1)==labels).float().mean().item()*100

    model.train()
    adapt_one_batch(model, images, base_text, args.tau, args.topk, args.iters, args.lr, args.alpha, args.no_diag, args.conf_thresh)
    model.eval()
    with torch.no_grad():
        z_img_t = model.encode_image(images); z_img_t = z_img_t / z_img_t.norm(dim=1, keepdim=True)
        tta_logits = z_img_t @ base_text.T
    tta_acc = (tta_logits.argmax(1)==labels).float().mean().item()*100

    print(f"[Zero-shot] {zs_acc:.2f}%   [CLIPArTT] {tta_acc:.2f}%")
    show_bar(zs_acc, tta_acc, args.corruption, args.severity)
    show_grid(images, zs_logits, tta_logits, labels, CIFAR10_CLASSES)

if __name__ == "__main__":
    main()
