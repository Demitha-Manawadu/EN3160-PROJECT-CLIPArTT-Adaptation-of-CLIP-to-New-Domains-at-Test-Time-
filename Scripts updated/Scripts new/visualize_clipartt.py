import os
import sys
import argparse
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

# Make "clip" importable when running from repo root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from clip import load, tokenize

# ---------------------- CIFAR-10-C minimal loader (no download) ----------------------
CIFAR10C_DIR_CANDIDATES = ["CIFAR-10-C", "cifar10-c"]
CIFAR10C_CORRUPTIONS = [
    "gaussian_noise","shot_noise","impulse_noise",
    "defocus_blur","glass_blur","motion_blur","zoom_blur",
    "snow","frost","fog","brightness","contrast",
    "elastic_transform","pixelate","jpeg_compression"
]
CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
TEMPLATE = "a photo of a {}"

def find_cifar10c_root(root_hint: str) -> str:
    base = os.path.expanduser(root_hint)
    for name in CIFAR10C_DIR_CANDIDATES:
        p = os.path.join(base, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "labels.npy")):
            return p
    raise FileNotFoundError(
        f"Couldn't find CIFAR-10-C under {base}. "
        f"Tried: {', '.join(os.path.join(base, n) for n in CIFAR10C_DIR_CANDIDATES)}"
    )

class CIFAR10C(Dataset):
    def __init__(self, root: str, corruption: str, severity: int, transform=None):
        assert corruption in CIFAR10C_CORRUPTIONS
        assert 1 <= severity <= 5
        base = find_cifar10c_root(root)
        self.transform = transform

        data_path   = os.path.join(base, f"{corruption}.npy")
        labels_path = os.path.join(base, "labels.npy")
        if not (os.path.isfile(data_path) and os.path.isfile(labels_path)):
            raise FileNotFoundError(f"Missing {corruption}.npy or labels.npy in {base}")

        imgs   = np.load(data_path)      # (50000, 32, 32, 3)
        labels = np.load(labels_path)    # (50000,)
        start, end = (severity-1)*10000, severity*10000
        self.imgs = imgs[start:end]
        self.labels = labels[start:end]

    def __len__(self): return 10000
    def __getitem__(self, i):
        img = Image.fromarray(self.imgs[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])

# ---------------------- CLIP/CLIPArTT helpers ----------------------
def set_visual_ln_trainable(model: nn.Module) -> List[nn.Parameter]:
    # make only visual LayerNorm affine parameters trainable
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for _, m in model.visual.named_modules():
        if isinstance(m, nn.LayerNorm) and getattr(m, "elementwise_affine", True):
            if getattr(m, "weight", None) is not None:
                m.weight.requires_grad_(True); params.append(m.weight)
            if getattr(m, "bias", None) is not None:
                m.bias.requires_grad_(True); params.append(m.bias)
    return params

@torch.no_grad()
def build_base_text_features(model, device, classnames: List[str]) -> torch.Tensor:
    prompts = [TEMPLATE.format(c.replace("_"," ").strip()) for c in classnames]
    text = tokenize(prompts).to(device)
    zt = model.encode_text(text)
    zt = zt / zt.norm(dim=1, keepdim=True)
    return zt  # [C, D]

def column_softmax(x: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.softmax((x / tau).T, dim=1).T

def build_instance_prompts_from_topk(logits_img_txt: torch.Tensor, classnames: List[str], k: int) -> List[str]:
    topk = torch.topk(logits_img_txt, k=k, dim=1).indices
    prompts = []
    for i in range(topk.size(0)):
        names = [classnames[idx] for idx in topk[i].tolist()]
        prompts.append(f"a photo of a {' or '.join(names)}")
    return prompts

def adapt_one_batch(
    model, images, base_text_features, classnames,
    *, tau: float, k: int, iters: int, lr: float,
    alpha: float = 0.5, no_diag: bool = True, conf_thresh: float = 0.3
):
    """
    Same inner loop you used:
    - Top-K from base prompts
    - pseudo-labels Q from alpha*Sv + (1-alpha)*St (column-softmax)
    - cross-entropy between Q and row-softmax(z_img @ zt_hat^T / tau)
    """
    device = images.device
    eps = 1e-6
    params = set_visual_ln_trainable(model)
    optim_ = optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)

    for _ in range(iters):
        z_img = model.encode_image(images)
        z_img = z_img / (z_img.norm(dim=1, keepdim=True) + eps)

        logits_base = (z_img @ base_text_features.T)                # [B, C]
        inst_prompts = build_instance_prompts_from_topk(logits_base, classnames, k)
        tokens = tokenize(inst_prompts).to(device)
        zt_hat = model.encode_text(tokens)
        zt_hat = zt_hat / (zt_hat.norm(dim=1, keepdim=True) + eps)

        S_v = z_img @ z_img.T
        S_t = zt_hat @ zt_hat.T
        S   = alpha * S_v + (1.0 - alpha) * S_t
        if no_diag:
            S = S - torch.diag_embed(torch.diag(S))
        Q = column_softmax(S, tau=tau)                              # [B, B]

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

        optim_.zero_grad(set_to_none=True)
        loss.backward()
        optim_.step()

# ---------------------- Visualization ----------------------
def imgrid_with_preds(
    images_tensor, zs_logits, tta_logits, classnames: List[str],
    ncols=4, save_path=None, title=None
):
    """
    images_tensor: [B, 3, H, W] (already *preprocessed*; so we need to approx undo normalize for display)
    We'll run a light un-normalize assuming CLIP preprocess (mean/std).
    """
    import math

    B = images_tensor.size(0)
    nrows = math.ceil(B / ncols)

    # CLIP vit-b/32 mean/std (OpenAI): (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    mean = torch.tensor([0.48145466, 0.4578275 , 0.40821073]).view(1,3,1,1).to(images_tensor.device)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1).to(images_tensor.device)
    imgs_disp = (images_tensor * std + mean).clamp(0,1).cpu()

    zs_prob  = torch.softmax(zs_logits,  dim=1)
    tta_prob = torch.softmax(tta_logits, dim=1)
    zs_pred  = zs_prob.argmax(dim=1).cpu().tolist()
    tta_pred = tta_prob.argmax(dim=1).cpu().tolist()
    zs_conf  = zs_prob.max(dim=1).values.cpu().tolist()
    tta_conf = tta_prob.max(dim=1).values.cpu().tolist()

    plt.figure(figsize=(3.2*ncols, 3.6*nrows))
    if title:
        plt.suptitle(title, fontsize=14)

    for i in range(B):
        ax = plt.subplot(nrows, ncols, i+1)
        img = imgs_disp[i].permute(1,2,0).numpy()
        ax.imshow(img)
        ax.axis("off")
        zs_txt  = f"ZS:  {classnames[zs_pred[i]]} ({zs_conf[i]*100:.1f}%)"
        tta_txt = f"TTA: {classnames[tta_pred[i]]} ({tta_conf[i]*100:.1f}%)"
        ax.set_title(zs_txt + "\n" + tta_txt, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()

def bar_compare(zs_acc: float, tta_acc: float, save_path=None, title=None):
    plt.figure(figsize=(4.5, 4))
    plt.bar(["Zero-shot", "CLIPArTT"], [zs_acc, tta_acc])
    plt.ylim(0, 100)
    if title:
        plt.title(title)
    for i, v in enumerate([zs_acc, tta_acc]):
        plt.text(i, v + 1.5, f"{v:.2f}%", ha="center", fontsize=10)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser("Visualize CLIPArTT improvement on CIFAR-10-C")
    ap.add_argument("--cifar-c-root", type=str, default="~/.torch/data")
    ap.add_argument("--corruption", type=str, default="glass_blur", choices=CIFAR10C_CORRUPTIONS)
    ap.add_argument("--severity", type=int, default=5, choices=[1,2,3,4,5])
    ap.add_argument("--model", type=str, default="ViT-B/32")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n", type=int, default=16, help="how many images to visualize (<= batch-size)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--no-diag", action="store_true", default=True)
    ap.add_argument("--conf-thresh", type=float, default=0.3)
    ap.add_argument("--save-dir", type=str, default="viz_out")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load CLIP
    model, preprocess = load(args.model, device=device)

    # Dataset & one batch
    ds = CIFAR10C(root=args.cifar_c_root, corruption=args.corruption, severity=args.severity, transform=preprocess)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    images, labels = next(iter(loader))
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    # Build base text features
    base_text = build_base_text_features(model, device, CIFAR10_CLASSES)

    # Zero-shot logits on this batch
    with torch.no_grad():
        z_img = model.encode_image(images)
        z_img = z_img / z_img.norm(dim=1, keepdim=True)
        zs_logits = z_img @ base_text.T
        zs_pred = zs_logits.argmax(dim=1)
        zs_acc_batch = (zs_pred == labels).float().mean().item() * 100.0

    # TTA on this same batch
    model.train()
    adapt_one_batch(
        model, images, base_text, CIFAR10_CLASSES,
        tau=args.tau, k=args.topk, iters=args.iters, lr=args.lr,
        alpha=args.alpha, no_diag=args.no_diag, conf_thresh=args.conf_thresh
    )
    model.eval()
    with torch.no_grad():
        z_img_t = model.encode_image(images)
        z_img_t = z_img_t / z_img_t.norm(dim=1, keepdim=True)
        tta_logits = z_img_t @ base_text.T
        tta_pred = tta_logits.argmax(dim=1)
        tta_acc_batch = (tta_pred == labels).float().mean().item() * 100.0

    # BAR CHART
    title = f"{args.corruption} | severity={args.severity} | batch={args.batch_size} | iters={args.iters}, lr={args.lr}"
    bar_compare(
        zs_acc_batch, tta_acc_batch,
        save_path=os.path.join(args.save_dir, f"bars_{args.corruption}_s{args.severity}.png"),
        title=title
    )

    # IMAGE GRID (first n samples from this batch)
    n = min(args.n, images.size(0))
    imgrid_with_preds(
        images[:n], zs_logits[:n], tta_logits[:n], CIFAR10_CLASSES,
        ncols=4,
        save_path=os.path.join(args.save_dir, f"grid_{args.corruption}_s{args.severity}.png"),
        title=title
    )

    print(f"Saved images to: {os.path.abspath(args.save_dir)}")
    print(f"Batch zero-shot acc: {zs_acc_batch:.2f}% | TTA acc: {tta_acc_batch:.2f}%")

if __name__ == "__main__":
    main()
