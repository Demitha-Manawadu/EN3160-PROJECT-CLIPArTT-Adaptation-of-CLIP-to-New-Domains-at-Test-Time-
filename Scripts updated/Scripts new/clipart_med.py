import os
import sys
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST
from tqdm import tqdm

# Make "clip" importable when running from repo root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from clip import load, tokenize

# ---------------------- Defaults ----------------------
DEFAULT_TOPK = 3
DEFAULT_TAU = 0.01
DEFAULT_ITERS = 15
DEFAULT_LR = 5e-4
DEFAULT_BS = 128
DEFAULT_WORKERS = 4
TEMPLATE = "a photo of a {}"

# CLIP mean/std (don’t use 0.5/0.5 – use CLIP’s stats)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# ---------------------- Noise Aug ----------------------
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        # tensor expected in [0,1] before Normalize
        noise = torch.randn_like(tensor) * self.std + self.mean
        return (tensor + noise).clamp(0.0, 1.0)

# ---------------------- CLIP helpers ----------------------
def set_visual_ln_trainable(model: nn.Module) -> List[nn.Parameter]:
    """Enable grads only for visual LayerNorm affine params."""
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for _, m in model.visual.named_modules():
        if isinstance(m, nn.LayerNorm) and getattr(m, "elementwise_affine", True):
            if getattr(m, "weight", None) is not None:
                m.weight.requires_grad_(True); trainable.append(m.weight)
            if getattr(m, "bias", None) is not None:
                m.bias.requires_grad_(True); trainable.append(m.bias)
    return trainable

@torch.no_grad()
def build_base_text_features(model, device, classnames: List[str]) -> torch.Tensor:
    prompts = [TEMPLATE.format(c.replace("_", " ").strip()) for c in classnames]
    text = tokenize(prompts).to(device)
    zt = model.encode_text(text)
    zt = zt / zt.norm(dim=1, keepdim=True)
    return zt  # [C, D]

def column_softmax(x: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.softmax((x / tau).T, dim=1).T

def build_instance_prompts_from_topk(logits_img_txt: torch.Tensor, classnames: List[str], k: int) -> List[str]:
    k = max(1, min(k, len(classnames)))
    topk = torch.topk(logits_img_txt, k=k, dim=1).indices
    prompts = []
    for i in range(topk.size(0)):
        names = [classnames[idx] for idx in topk[i].tolist()]
        prompts.append(f"a photo of a {' or '.join(names)}")
    return prompts

# ---------------------- CLIPArTT core ----------------------
def clipartt_adapt_batch(
    model,
    images: torch.Tensor,
    base_text_features: torch.Tensor,
    classnames: List[str],
    *,
    tau: float = DEFAULT_TAU,
    K: int = DEFAULT_TOPK,
    iters: int = DEFAULT_ITERS,
    optimizer: optim.Optimizer = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = images.device
    if optimizer is None:
        params = set_visual_ln_trainable(model)
        optimizer = optim.Adam(params, lr=DEFAULT_LR)

    for _ in range(iters):
        z_img = model.encode_image(images)
        z_img = z_img / z_img.norm(dim=1, keepdim=True)

        logits_base = (z_img @ base_text_features.T) / tau
        inst_prompts = build_instance_prompts_from_topk(logits_base, classnames, K)
        tokens = tokenize(inst_prompts).to(device)
        zt_hat = model.encode_text(tokens)
        zt_hat = zt_hat / zt_hat.norm(dim=1, keepdim=True)

        S_v = z_img @ z_img.T
        S_t = zt_hat @ zt_hat.T
        Q = column_softmax((S_v + S_t) / 2.0, tau=tau)

        inter_logits = torch.log_softmax((z_img @ zt_hat.T) / tau, dim=1)
        loss = -(Q * inter_logits).sum(dim=0).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        z_img = model.encode_image(images)
        z_img = z_img / z_img.norm(dim=1, keepdim=True)
        logits_base = (z_img @ base_text_features.T)

    return z_img, logits_base

# ---------------------- Evaluation ----------------------
@torch.no_grad()
def evaluate_zero_shot_med(model, loader, base_text_features, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in tqdm(loader, desc="Zero-shot evaluation"):
        images = images.to(device)
        labels = labels.squeeze().long().to(device)   # <-- FIX: squeeze to [B]
        z = model.encode_image(images)
        z = z / z.norm(dim=1, keepdim=True)
        logits = z @ base_text_features.T
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0

def evaluate_with_clipartt_med(model, loader, base_text_features, classnames, device, args):
    params = set_visual_ln_trainable(model)
    optimizer = optim.Adam(params, lr=args.lr)

    correct, total = 0, 0
    for images, labels in tqdm(loader, desc="TTA evaluation"):
        images = images.to(device)
        labels = labels.squeeze().long().to(device)   # <-- FIX: squeeze to [B]

        _, logits = clipartt_adapt_batch(
            model,
            images,
            base_text_features,
            classnames,
            tau=args.tau,
            K=min(args.topk, len(classnames)),
            iters=args.iters,
            optimizer=optimizer,
        )
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total if total > 0 else 0.0

# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser(description="CLIPArTT TTA on PathMNIST")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--severity", type=int, default=3, help="Gaussian noise severity (1-5)")
    parser.add_argument("--noise-std", type=float, default=None, help="Override Gaussian noise std (0..1)")
    parser.add_argument("--model", type=str, default="ViT-B/32")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Map severity to noise std (optional)
    default_std_map = {1:0.05, 2:0.10, 3:0.15, 4:0.20, 5:0.25}
    std_dev = args.noise_std if args.noise_std is not None else default_std_map.get(args.severity, 0.15)

    # CLIP-style preprocess with optional Gaussian noise BEFORE Normalize
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),                 # PathMNIST is RGB 28x28; ToTensor -> [0,1]
        AddGaussianNoise(std=std_dev),         # add synthetic corruption (optional)
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    print("Starting dataset download if necessary...")
    ds = PathMNIST(split=args.split, transform=preprocess, download=True)
    # Build classnames from dataset info (PathMNIST has 9 classes)
    # info['label'] is typically a dict mapping class index (as str) -> name
    label_dict = ds.info.get("label", None)
    if label_dict is None:
        # Fallback to generic names if unavailable
        classnames = [f"class_{i}" for i in range(int(ds.info.get("n_classes", 9)))]
    else:
        # Keys may be strings "0","1",... Ensure index order
        # Each value can be a string or list; we take the string
        indices = sorted([int(k) for k in label_dict.keys()])
        classnames = []
        for i in indices:
            name = label_dict[str(i)]
            if isinstance(name, (list, tuple)):
                name = name[0]
            classnames.append(str(name))
    print(f"Classes ({len(classnames)}): {classnames}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model, _ = load(args.model, device=device)
    base_text_features = build_base_text_features(model, device, classnames)

    zs_acc = evaluate_zero_shot_med(model, loader, base_text_features, device)
    print(f"[Zero-shot] Accuracy: {zs_acc:.2f}% using {args.model}")

    tta_acc = evaluate_with_clipartt_med(model, loader, base_text_features, classnames, device, args)
    print(f"[CLIPArTT] Accuracy after TTA: {tta_acc:.2f}% | "
          f"K={min(args.topk, len(classnames))}, tau={args.tau}, iters={args.iters}, "
          f"lr={args.lr}, batch_size={args.batch_size}, noise_std={std_dev}")

if __name__ == "__main__":
    main()
