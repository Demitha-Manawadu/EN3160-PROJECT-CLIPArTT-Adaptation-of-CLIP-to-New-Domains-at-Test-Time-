# scripts/visualize_clipartt_med.py
import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from medmnist import PathMNIST

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
DEFAULT_N = 16
TEMPLATE = "a photo of a {}"

# CLIP mean/std
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

# ---------------------- Eval helpers ----------------------
@torch.no_grad()
def run_zero_shot_preds(model, loader, base_text_features, device):
    model.eval()
    preds, targets = [], []
    for images, labels in tqdm(loader, desc="Zero-shot pass"):
        images = images.to(device)
        labels = labels.squeeze().long().to(device)
        z = model.encode_image(images)
        z = z / z.norm(dim=1, keepdim=True)
        logits = z @ base_text_features.T
        preds.append(logits.argmax(dim=1).cpu())
        targets.append(labels.cpu())
    preds = torch.cat(preds, 0)
    targets = torch.cat(targets, 0)
    acc = 100.0 * (preds == targets).float().mean().item()
    return preds.numpy(), targets.numpy(), acc

def run_tta_preds(model, loader, base_text_features, classnames, device, args):
    params = set_visual_ln_trainable(model)
    optimizer = optim.Adam(params, lr=args.lr)

    model.train()
    preds = []
    targets = []
    for images, labels in tqdm(loader, desc="CLIPArTT pass"):
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        _, logits = clipartt_adapt_batch(
            model, images, base_text_features, classnames,
            tau=args.tau, K=min(args.topk, len(classnames)), iters=args.iters,
            optimizer=optimizer
        )
        preds.append(logits.argmax(dim=1).cpu())
        targets.append(labels.cpu())

    preds = torch.cat(preds, 0).numpy()
    targets = torch.cat(targets, 0).numpy()
    acc = 100.0 * (preds == targets).mean().item()
    return preds, targets, acc

# ---------------------- Visualization ----------------------
def save_sample_grid(
    ds_raw, idxs, classnames, zs_preds, tta_preds, targets,
    outpath, cols=4
):
    import torch
    import numpy as np
    from torchvision import transforms

    rows = int(np.ceil(len(idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*3.2))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    to_pil = transforms.ToPILImage()

    for j, idx in enumerate(idxs):
        r, c = divmod(j, cols)
        ax = axes[r, c]

        # ds_raw returns (PIL.Image or ndarray, label)
        img_raw, _ = ds_raw[idx]
        # normalize to PIL for display
        if isinstance(img_raw, torch.Tensor):
            img_disp = to_pil(img_raw)
        elif isinstance(img_raw, np.ndarray):
            if img_raw.ndim == 2:
                img_disp = to_pil(torch.from_numpy(img_raw))
            else:
                img_disp = to_pil(torch.from_numpy(np.transpose(img_raw, (2,0,1))))
        else:
            img_disp = img_raw  # already PIL

        ax.imshow(img_disp)
        gt    = classnames[int(targets[idx])]
        zsp   = classnames[int(zs_preds[idx])]
        ttap  = classnames[int(tta_preds[idx])]
        ok_zs = "✓" if zs_preds[idx] == targets[idx] else "✗"
        ok_tt = "✓" if tta_preds[idx] == targets[idx] else "✗"
        ax.set_title(f"GT: {gt}\nZS: {zsp} {ok_zs} | TTA: {ttap} {ok_tt}", fontsize=9)
        ax.axis('off')

    # hide empties
    for k in range(len(idxs), rows*cols):
        r, c = divmod(k, cols)
        axes[r, c].axis('off')

    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=180)
    plt.close(fig)

# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize CLIPArTT on PathMNIST")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--severity", type=int, default=3, help="Gaussian noise severity (1-5)")
    parser.add_argument("--noise-std", type=float, default=None, help="Override Gaussian noise std (0..1)")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="number of samples to show")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="ViT-B/32")
    parser.add_argument("--out", type=str, default="outputs/clipartt_pathmnist_grid.png")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # map severity to noise std
    default_std_map = {1:0.05, 2:0.10, 3:0.15, 4:0.20, 5:0.25}
    std_dev = args.noise_std if args.noise_std is not None else default_std_map.get(args.severity, 0.15)

    # Preprocess for the model
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        AddGaussianNoise(std=std_dev),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    # Raw dataset (NO transform) for display
    ds_raw = PathMNIST(split=args.split, transform=None, download=True)

    # Classnames from dataset info (PathMNIST has 9 classes)
    label_dict = ds_raw.info.get("label", None)
    if label_dict is None:
        classnames = [f"class_{i}" for i in range(int(ds_raw.info.get("n_classes", 9)))]
    else:
        indices = sorted([int(k) for k in label_dict.keys()])
        classnames = []
        for i in indices:
            name = label_dict[str(i)]
            if isinstance(name, (list, tuple)):
                name = name[0]
            classnames.append(str(name))
    print(f"Classes ({len(classnames)}): {classnames}")

    # Same dataset for model input with transforms
    ds_vis = PathMNIST(split=args.split, transform=preprocess, download=True)
    loader = DataLoader(ds_vis, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model, _ = load(args.model, device=device)
    base_text_features = build_base_text_features(model, device, classnames)

    print("Running zero-shot…")
    zs_preds, targets, zs_acc = run_zero_shot_preds(model, loader, base_text_features, device)
    print(f"[Zero-shot] Accuracy: {zs_acc:.2f}%")

    print("Running CLIPArTT (TTA)…")
    tta_preds, targets2, tta_acc = run_tta_preds(model, loader, base_text_features, classnames, device, args)
    assert np.array_equal(targets, targets2)
    print(f"[CLIPArTT] Accuracy: {tta_acc:.2f}% | K={min(args.topk, len(classnames))}, tau={args.tau}, iters={args.iters}, lr={args.lr}")

    # Choose indices where ZS != TTA (to visualize change). Fill if not enough.
    n = min(args.n, len(ds_vis))
    idx_pool = np.arange(len(ds_vis))
    diffs = np.where(zs_preds != tta_preds)[0]
    chosen = diffs.tolist()[:n]
    if len(chosen) < n:
        extra = np.setdiff1d(idx_pool, np.array(chosen, dtype=int), assume_unique=False)
        if len(extra) > 0:
            fill = rng.choice(extra, size=n-len(chosen), replace=False).tolist()
            chosen += fill

    save_sample_grid(
        ds_raw=ds_raw,
        idxs=chosen,
        classnames=classnames,
        zs_preds=zs_preds,
        tta_preds=tta_preds,
        targets=targets,
        outpath=args.out,
        cols=4
    )
    print(f"Saved grid to: {args.out}")

if __name__ == "__main__":
    main()
