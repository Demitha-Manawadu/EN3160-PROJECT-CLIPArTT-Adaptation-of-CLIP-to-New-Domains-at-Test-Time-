import os
import sys
import math
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

# Make "clip" importable when running from repo root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from clip import load, tokenize

# ========== Paper defaults ==========
DEFAULT_K = 3          # top-K classes to build instance-specific prompts (Sec. 3.2)
DEFAULT_TAU = 0.01     # temperature for softmax on similarities (Eq. 3 & 4)
DEFAULT_ITERS = 10     # adaptation steps per incoming batch (Table 3)
DEFAULT_LR = 1e-3      # Adam on visual LayerNorm only (Sec. 4 Test-time adaptation)
DEFAULT_BS = 128
DEFAULT_WORKERS = 4
TEMPLATE = "a photo of a {}"

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"
]
CIFAR100_CLASSES = None  # will read from dataset


# ---------------------- Helpers ----------------------

def set_visual_ln_trainable(model: nn.Module) -> List[nn.Parameter]:
    """
    Enable grads ONLY for LayerNorm params in the VISUAL encoder, as in CLIPArTT.
    Freeze everything else (including the text encoder).
    """
    # Freeze all
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Enable grads on visual LayerNorms (weight + bias)
    trainable = []
    for name, m in model.visual.named_modules():
        if isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:
                m.weight.requires_grad_(True)
                m.bias.requires_grad_(True)
                trainable += [m.weight, m.bias]
    model.train()
    return trainable


@torch.no_grad()
def build_base_text_features(model, device, classnames: List[str]) -> torch.Tensor:
    prompts = [TEMPLATE.format(c.replace("_", " ").strip()) for c in classnames]
    text = tokenize(prompts).to(device)
    zt = model.encode_text(text)
    zt = zt / zt.norm(dim=1, keepdim=True)
    return zt  # [K, D]


def column_softmax(x: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Apply softmax column-wise: for each column j, softmax over i.
    Equivalent to softmax on x^T along dim=1 and transpose back.
    """
    return torch.softmax((x / tau).T, dim=1).T


def build_instance_prompts_from_topk(
    logits_img_txt: torch.Tensor,
    classnames: List[str],
    K: int
) -> List[str]:
    """
    For each image i, take top-K class names and produce:
        "a photo of a {c1} or {c2} or {c3}"
    (Sec. 3.2: instance-specific multi-class prompt)
    """
    topk = torch.topk(logits_img_txt, k=K, dim=1).indices  # [B, K]
    prompts = []
    for i in range(topk.size(0)):
        names = [classnames[idx] for idx in topk[i].tolist()]
        joined = " or ".join(names)
        prompts.append(f"a photo of a {joined}")
    return prompts


def cross_entropy_with_soft_targets(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred: probabilities P_hat (B x B)
    target: soft labels Q (B x B), column-stochastic
    CE averaged over columns (Eq. 5).
    """
    eps = 1e-12
    loss = -(target * (pred + eps).log()).sum(dim=0).mean()
    return loss


# ---------------------- CLIPArTT core ----------------------

def clipartt_adapt_batch(
    model,
    images: torch.Tensor,
    base_text_features: torch.Tensor,
    classnames: List[str],
    *,
    tau: float = DEFAULT_TAU,
    K: int = DEFAULT_K,
    iters: int = DEFAULT_ITERS,
    optimizer: optim.Optimizer = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs CLIPArTT TTA on a single batch (transductive).
    Returns:
        image_features (after last iter, L2-normed),
        base_logits (image-to-baseText logits for final evaluation)
    """
    device = images.device

    # Recompute optimizer per call if not provided
    if optimizer is None:
        params = set_visual_ln_trainable(model)
        optimizer = optim.Adam(params, lr=DEFAULT_LR)

    for _ in range(iters):
        # ---- Forward pass (recompute each iter) ----
        z_img = model.encode_image(images)                   # [B, D]
        z_img = z_img / z_img.norm(dim=1, keepdim=True)

        # Step 1: zero-shot over base class prompts -> to pick top-K (Eq. 1)
        logits_base = (z_img @ base_text_features.T) / tau   # [B, K_classes]

        # Build instance-specific text prompts from top-K
        inst_prompts = build_instance_prompts_from_topk(logits_base, classnames, K=K)
        tokens = tokenize(inst_prompts).to(device)
        zt_hat = model.encode_text(tokens)                   # [B, D]
        zt_hat = zt_hat / zt_hat.norm(dim=1, keepdim=True)

        # Step 2: pairwise similarities & pseudo-labels Q (Eq. 3)
        S_v = z_img @ z_img.T                                # [B, B] cosine since L2-normed
        S_t = zt_hat @ zt_hat.T                              # [B, B]
        Q = column_softmax((S_v + S_t) / 2.0, tau=tau)       # col-wise softmax

        # Step 3: prediction matrix P̂ via image-to-text with instance prompts (Eq. 4)
        P_hat = torch.softmax((z_img @ zt_hat.T) / tau, dim=1)  # row-wise softmax

        # Step 4: CLIPArTT loss (Eq. 5)
        loss = cross_entropy_with_soft_targets(P_hat, Q)

        # Optimize only visual LN params
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Final features after adaptation
    with torch.no_grad():
        z_img = model.encode_image(images)
        z_img = z_img / z_img.norm(dim=1, keepdim=True)
        logits_base = (z_img @ base_text_features.T)

    return z_img, logits_base


# ---------------------- Data & Runner ----------------------

def get_dataset(args):
    if args.dataset.lower() == "cifar10":
        ds = datasets.CIFAR10(
            root=os.path.expanduser("~/.torch/data"),
            train=(args.split == "train"),
            download=True,
            transform=args.preprocess,
        )
        classnames = ds.classes if hasattr(ds, "classes") else CIFAR10_CLASSES
    elif args.dataset.lower() == "cifar100":
        ds = datasets.CIFAR100(
            root=os.path.expanduser("~/.torch/data"),
            train=(args.split == "train"),
            download=True,
            transform=args.preprocess,
        )
        classnames = ds.classes
    elif args.dataset.lower() == "imagefolder":
        assert args.root is not None, "For imagefolder, --root must be provided"
        ds = datasets.ImageFolder(root=args.root, transform=args.preprocess)
        classnames = ds.classes
    else:
        raise ValueError("dataset must be one of: cifar10 | cifar100 | imagefolder")
    return ds, classnames


def evaluate_zero_shot(model, loader, base_text_features, device):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            z = model.encode_image(images)
            z = z / z.norm(dim=1, keepdim=True)
            logits = z @ base_text_features.T
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def evaluate_with_clipartt(model, loader, base_text_features, classnames, device, args):
    params = set_visual_ln_trainable(model)
    optimizer = optim.Adam(params, lr=args.lr)

    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # CLIPArTT TTA on this batch (transductive)
        _, logits = clipartt_adapt_batch(
            model,
            images,
            base_text_features,
            classnames,
            tau=args.tau,
            K=args.topk,
            iters=args.iters,
            optimizer=optimizer,
        )

        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


def main():
    p = argparse.ArgumentParser("CLIPArTT – Test-Time Adaptation for CLIP")
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "cifar100", "imagefolder"])
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--root", type=str, default=None, help="root for ImageFolder")
    p.add_argument("--model", type=str, default="ViT-B/32")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BS)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--topk", type=int, default=DEFAULT_K)
    p.add_argument("--tau", type=float, default=DEFAULT_TAU)
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load CLIP
    model, preprocess = load(args.model, device=device)
    args.preprocess = preprocess

    # Dataset & loader
    ds, classnames = get_dataset(args)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # Base (single-class) text embeddings used for evaluation logits (Eq. 1)
    base_text_features = build_base_text_features(model, device, classnames)

    # Baseline zero-shot accuracy
    zs_acc = evaluate_zero_shot(model, loader, base_text_features, device)
    print(f"[Zero-shot] Accuracy: {zs_acc:.2f}% using {args.model}")

    # CLIPArTT TTA accuracy
    tta_acc = evaluate_with_clipartt(model, loader, base_text_features, classnames, device, args)
    print(f"[CLIPArTT] Accuracy after TTA: {tta_acc:.2f}% | "
          f"K={args.topk}, tau={args.tau}, iters={args.iters}, lr={args.lr}, bs={args.batch-size}")


if __name__ == "__main__":
    main()
