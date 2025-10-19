import os
import sys
import time
import argparse
import tarfile
import hashlib
import urllib.request
from typing import List, Tuple, Dict, Iterable

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Make "clip" importable when running from repo root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from clip import load, tokenize

# ---------------------- Defaults ----------------------
DEFAULT_K = 5              # wider top-k (robust mode)
DEFAULT_TAU = 0.07         # smoother softmax (robust mode)
DEFAULT_ITERS = 5
DEFAULT_LR = 5e-4
DEFAULT_BS = 128
DEFAULT_WORKERS = 4
GRAD_CLIP_NORM = 1.0
CONF_THRESH = 0.40         # mask weak columns (robust mode)
TEMPLATE = "a photo of a {}"

# ---------------------- CIFAR-10-C ----------------------
CIFAR10C_URL = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"
CIFAR10C_MD5 = "56bf5dcef84df0e2308c6dcbcbbd8499"
CIFAR10C_DIR = "CIFAR-10-C"
CIFAR10C_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression"
]
CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# ---------------------- Helpers ----------------------
# detect both nn.LayerNorm and CLIP's custom LayerNorm
try:
    from clip.model import LayerNorm as ClipLayerNorm  # type: ignore
    LN_TYPES = (nn.LayerNorm, ClipLayerNorm)
except Exception:
    LN_TYPES = (nn.LayerNorm,)

def set_visual_ln_trainable(model: nn.Module) -> List[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad_(False)
    params: List[nn.Parameter] = []
    for _, m in model.visual.named_modules():
        if isinstance(m, LN_TYPES) and getattr(m, "elementwise_affine", True):
            if hasattr(m, "weight") and m.weight is not None:
                m.weight.requires_grad_(True); params.append(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad_(True);   params.append(m.bias)
    return params

def snapshot_visual_ln(model: nn.Module):
    snaps = []
    for _, m in model.visual.named_modules():
        if isinstance(m, LN_TYPES) and getattr(m, "elementwise_affine", True):
            if hasattr(m, "weight") and m.weight is not None:
                snaps.append((m.weight, m.weight.detach().clone()))
            if hasattr(m, "bias") and m.bias is not None:
                snaps.append((m.bias,   m.bias.detach().clone()))
    return snaps

def restore_visual_ln(snaps):
    for p, v in snaps:
        p.data.copy_(v)

@torch.no_grad()
def build_base_text_features(model, device, classnames: List[str]) -> torch.Tensor:
    prompts = [TEMPLATE.format(c.replace("_", " ").strip()) for c in classnames]
    text = tokenize(prompts).to(device)
    zt = model.encode_text(text)
    zt = zt / zt.norm(dim=1, keepdim=True)
    return zt  # [C, D]

def column_softmax(x: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.softmax((x / tau).T, dim=1).T

def build_instance_prompts_from_topk(logits: torch.Tensor, classnames: List[str], K: int) -> List[str]:
    topk = torch.topk(logits, k=K, dim=1).indices
    prompts = []
    for i in range(topk.size(0)):
        names = [classnames[idx] for idx in topk[i].tolist()]
        prompts.append(f"a photo of a {' or '.join(names)}")
    return prompts

# ---------------------- CIFAR-10-C fetch ----------------------
def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _download_with_retries(url: str, dst: str, max_tries: int = 8) -> None:
    backoff = 5
    for i in range(1, max_tries + 1):
        try:
            print(f"Downloading attempt {i}/{max_tries} -> {dst}")
            urllib.request.urlretrieve(url, dst)
            return
        except Exception as e:
            if i == max_tries: raise
            print(f"Download error: {e}. Retrying in {backoff} s")
            time.sleep(backoff); backoff *= 2

def _safe_extract_tar(tar_path: str, out_dir: str) -> None:
    def is_within_directory(directory, target):
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        return os.path.commonprefix([abs_directory, abs_target]) == abs_directory
    with tarfile.open(tar_path, "r") as tf:
        for m in tf.getmembers():
            target_path = os.path.join(out_dir, m.name)
            if not is_within_directory(out_dir, target_path):
                raise RuntimeError("Blocked path traversal in tar file")
        tf.extractall(path=out_dir)

def ensure_cifar10c(root: str):
    os.makedirs(root, exist_ok=True)
    tar_path = os.path.join(root, "CIFAR-10-C.tar")
    target_dir = os.path.join(root, CIFAR10C_DIR)

    if os.path.isdir(target_dir) and os.path.isfile(os.path.join(target_dir, "labels.npy")):
        return

    need_download = True
    if os.path.isfile(tar_path):
        try:
            if _md5(tar_path) == CIFAR10C_MD5:
                need_download = False
        except Exception:
            need_download = True

    if need_download:
        if os.path.isfile(tar_path):
            try: os.remove(tar_path)
            except Exception: pass
        print(f"Downloading CIFAR-10-C to {tar_path}")
        _download_with_retries(CIFAR10C_URL, tar_path)
        md5 = _md5(tar_path)
        if md5 != CIFAR10C_MD5:
            raise RuntimeError(f"MD5 mismatch for CIFAR-10-C.tar (got {md5}, expected {CIFAR10C_MD5})")

    print(f"Extracting {tar_path}")
    _safe_extract_tar(tar_path, root)
    print("Extraction complete")

# ---------------------- Dataset ----------------------
class CIFAR10C(Dataset):
    def __init__(self, root: str, severity: int, corruption: str, transform=None):
        assert severity in [1,2,3,4,5], "severity must be 1..5"
        base = os.path.join(root, CIFAR10C_DIR)
        labels_path = os.path.join(base, "labels.npy")
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(f"labels.npy not found at {labels_path}. Run ensure_cifar10c(root) first.")
        self.transform = transform
        self.labels_block = np.load(labels_path)  # (10000,)
        self.arr = np.load(os.path.join(base, f"{corruption}.npy"), mmap_mode="r")  # (50000, 32, 32, 3)
        self.off = (severity - 1) * 10000

    def __len__(self): return 10000
    def __getitem__(self, i):
        img_np = np.array(self.arr[self.off + i])
        label = int(self.labels_block[i])
        img = Image.fromarray(img_np)
        if self.transform is not None: img = self.transform(img)
        return img, label

# ---------------------- CLIPArTT (classic vs robust) ----------------------
def clipartt_adapt_batch(
    model,
    images: torch.Tensor,
    base_text_features: torch.Tensor,
    classnames: List[str],
    *,
    tau: float,
    K: int,
    iters: int,
    optimizer: optim.Optimizer,
    conf_thresh: float,
    use_logit_scale: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:

    device = images.device
    s = model.logit_scale.exp() if use_logit_scale else None

    for _ in range(iters):
        # Image features
        z_img = model.encode_image(images)                 # [B, D]
        z_img = z_img / z_img.norm(dim=1, keepdim=True)

        # Top-K class prompts
        logits_base = (z_img @ base_text_features.T) if s is None else (s * (z_img @ base_text_features.T))
        inst_prompts = build_instance_prompts_from_topk(logits_base / max(tau, 1e-6), classnames, K=K)

        # Instance-specific text
        tokens = tokenize(inst_prompts).to(device)
        zt_hat = model.encode_text(tokens)                 # [B, D]
        zt_hat = zt_hat / zt_hat.norm(dim=1, keepdim=True)

        # Soft pairwise targets
        S_v = z_img @ z_img.T
        S_t = zt_hat @ zt_hat.T
        Q = column_softmax((S_v + S_t) / 2.0, tau=tau)     # column-stochastic

        # Pred matrix (row-softmax)
        inter_logits = (z_img @ zt_hat.T) if s is None else (s * (z_img @ zt_hat.T))
        inter_logits = inter_logits / max(tau, 1e-6)
        log_probs = torch.log_softmax(inter_logits, dim=1)

        # Optional confidence masking over columns
        if conf_thresh > 0.0:
            with torch.no_grad():
                probs = torch.softmax(inter_logits, dim=1)
                col_conf = probs.max(dim=0).values
                col_mask = (col_conf >= conf_thresh).float()
                col_w = col_mask / (col_mask.sum() + 1e-6)
            loss = -((Q * log_probs).sum(dim=0) * col_w).sum()
        else:
            loss = -(Q * log_probs).sum(dim=0).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], GRAD_CLIP_NORM)
        optimizer.step()

    with torch.no_grad():
        z_img = model.encode_image(images); z_img = z_img / z_img.norm(dim=1, keepdim=True)
        final_logits = z_img @ base_text_features.T
    return z_img, final_logits

# ---------------------- Eval ----------------------
@torch.no_grad()
def evaluate_zero_shot(model, loader, base_text_features, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
        z = model.encode_image(images); z = z / z.norm(dim=1, keepdim=True)
        logits = z @ base_text_features.T
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item(); total += labels.size(0)
    return 100.0 * correct / total

def evaluate_with_clipartt(model, loader, base_text_features, classnames, device, args, use_logit_scale: bool):
    ln_snapshot = snapshot_visual_ln(model)
    correct, total = 0, 0
    for images, labels in loader:
        restore_visual_ln(ln_snapshot)
        images = images.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
        params = set_visual_ln_trainable(model)
        if len(params) == 0:
            print("[Warn] No trainable LayerNorm params found!")
        optimizer = optim.Adam(params, lr=args.lr)
        model.train()
        _, logits = clipartt_adapt_batch(
            model, images, base_text_features, classnames,
            tau=args.tau, K=args.topk, iters=args.iters, optimizer=optimizer,
            conf_thresh=args.conf_thresh, use_logit_scale=use_logit_scale,
        )
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item(); total += labels.size(0)
        model.eval()
    return 100.0 * correct / total

# ---------------------- Protocol ----------------------
def run_cifar10c_protocol(args, device, model, preprocess, use_logit_scale: bool):
    base_text = build_base_text_features(model, device, CIFAR10_CLASSES)

    def eval_one(cname: str, sev: int) -> Tuple[float, float]:
        ds = CIFAR10C(root=os.path.expanduser("~/.torch/data"), severity=sev, corruption=cname, transform=preprocess)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)
        zs = evaluate_zero_shot(model, loader, base_text, device)
        tta = evaluate_with_clipartt(model, loader, base_text, CIFAR10_CLASSES, device, args, use_logit_scale)
        return zs, tta

    corrs: Iterable[str] = CIFAR10C_CORRUPTIONS if args.corruption == "all" else [args.corruption]
    sevs: Iterable[int] = [args.severity]

    zs_list, tta_list = [], []
    for c in corrs:
        for s in sevs:
            print(f"\n==> corruption={c} | severity={s}")
            zs, tta = eval_one(c, s)
            tag = "CLIPArTT" if use_logit_scale else "Robust-CLIPArTT"
            print(f"[Zero-shot] {zs:.2f}%   [{tag}] {tta:.2f}%")
            zs_list.append(zs); tta_list.append(tta)

    zs_mean = float(np.mean(zs_list)); tta_mean = float(np.mean(tta_list))
    scope = ("all corruptions" if args.corruption == "all" else args.corruption) + f" @ severity {args.severity}"
    tag = "CLIPArTT" if use_logit_scale else "Robust-CLIPArTT"
    print(f"\n=== CIFAR-10-C mean over {scope} ===")
    print(f"[Zero-shot] {zs_mean:.2f}%   [{tag}] {tta_mean:.2f}%")

# ---------------------- Main ----------------------
def main():
    p = argparse.ArgumentParser("Robust/Classic CLIPArTT – CIFAR-10-C")
    p.add_argument("--model", type=str, default="ViT-B/32")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BS)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)

    # robust defaults (safe)
    p.add_argument("--topk", type=int, default=DEFAULT_K)
    p.add_argument("--tau", type=float, default=DEFAULT_TAU)
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--conf-thresh", type=float, default=CONF_THRESH)

    # dataset slice (you asked for severity=5 single/selected corruption)
    p.add_argument("--corruption", type=str, default="gaussian_noise",
                   help="one of the 15 names or 'all'")
    p.add_argument("--severity", type=int, default=5, choices=[1,2,3,4,5])

    # switch: classic paper behavior (uses logit_scale) vs robust
    p.add_argument("--classic", action="store_true",
                   help="use classic CLIPArTT inner loop (logit_scale on, K=3,tau=0.01, iters=10, lr=1e-3)")

    args = p.parse_args()

    # If classic mode: override to paper-ish settings
    use_logit_scale = False
    if args.classic:
        args.topk = 3
        args.tau = 0.01
        args.iters = 10
        args.lr = 1e-3
        args.conf_thresh = 0.0
        use_logit_scale = True

    # a few stability flags
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ensure_cifar10c(os.path.expanduser("~/.torch/data"))

    model, preprocess = load(args.model, device=device)

    run_cifar10c_protocol(args, device, model, preprocess, use_logit_scale)

if __name__ == "__main__":
    main()
