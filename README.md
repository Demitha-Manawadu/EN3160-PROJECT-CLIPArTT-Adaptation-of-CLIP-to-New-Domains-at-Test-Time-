# CLIPArTT – Adaptation of CLIP to New Domains at Test Time (WACV 2025 Reimplementation + Novel Application- Medical Domain Extension)

This repository implements and extends the **CLIPArTT** method – *“Adaptation of CLIP to New Domains at Test Time”* (WACV 2025) – which enables **test-time domain adaptation (TTA)** for CLIP models under distribution shifts, such as noisy or corrupted images.  

Our implementation faithfully reproduces the results reported in the paper for **CIFAR-10-C** and **CIFAR-100-C**, and extends the approach to the **medical imaging domain** (CheXpert / CheXphoto-style setups) to evaluate CLIP’s zero-shot robustness and adaptation under real-world corruptions.

---

##  Overview

###  What is CLIPArTT?
CLIPArTT (CLIP Adaptation at Test Time) performs **unsupervised, batch-wise adaptation** of the CLIP visual encoder to corrupted or shifted test data without retraining or labeled samples.  
It modifies **LayerNorm affine parameters** in CLIP’s vision transformer using self-consistency signals between image-image and text-text similarities.

<p align="center">
  <img src="https://raw.githubusercontent.com/openai/CLIP/main/CLIP.png" width="500"><br>
  <em>CLIP jointly aligns visual and textual representations through cosine similarity.</em>
</p>

---



##  Method Summary

CLIPArTT leverages self-consistency across test samples:
1. **Top-K pseudo labels** are generated using CLIP’s zero-shot text prompts.
2. **Image-image** (`S_v`) and **text-text** (`S_t`) similarity matrices are computed within each batch.
3. A **soft pseudo-label matrix** `Q = softmax((S_v + S_t) / 2τ)` is formed.
4. LayerNorm parameters in CLIP’s visual encoder are adapted by minimizing:
   \[
   \mathcal{L}_{TTA} = - \frac{1}{B}\sum_i Q_i^\top \log P_i
   \]
   where \(P_i\) are cosine-similarity logits between images and instance-specific text prompts.

---

## Datasets & Results

### **CIFAR-10-C / CIFAR-100-C**
We evaluate CLIPArTT on standard corrupted benchmarks.  
Each dataset contains 15 corruption types × 5 severity levels.

| Dataset | Zero-shot CLIP | CLIPArTT (ours) | Δ Improvement |
|----------|--------|----------------|------------------|---------------|
| CIFAR-10 (clean) | 88.8 | **88.9** | +0.1 |
| CIFAR-10-C (Impulse Noise, s=5) |  51.71 | **64.99** | +11.88|
| CIFAR-100-C (Impulse Noise, s=5) |  23.87 | **36.39** | +12.52 |

> ⚙️ Results reproduced using `ViT-B/32` and batch size = 128.  
> Accuracy improvements are consistent across most corruption types (Gaussian, Defocus, Glass, etc.).

---

# generate_readme_medical.py
content = """# Extension to Medical Images

To explore CLIP’s robustness in **medical imaging**, we extended CLIPArTT to a completely new domain beyond natural images.  
This experiment evaluates how well CLIP can adapt to **histopathological data** when domain shifts occur.

## Dataset: PathMNIST
| Aspect | Description |
|--------|-------------|
| **Origin** | Derived from the NCT-CRC-HE-100K colorectal cancer histology dataset. |
| **Classes** | 9 tissue types, including *normal mucosa*, *adenocarcinoma epithelium*, *lymphocytes*, *mucus*, *stroma*, and others. |
| **Task** | Multi-class classification of small pathology patches (28×28 RGB). |
| **Split** | 90% training + 10% validation (from NCT-CRC-HE-100K) and an external test set (CRC-VAL-HE-7K). |
| **Purpose** | To simulate medical domain shifts caused by changes in hospitals, scanners, or imaging conditions. |

---

## Corruption Setup
Unlike CIFAR-10-C, PathMNIST does not include predefined corruptions.  
To simulate domain degradation, we introduced **synthetic Gaussian noise** directly in the image preprocessing stage:

\`\`\`python
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return (tensor + noise).clamp(0.0, 1.0)
\`\`\`

The noise level (standard deviation) was controlled by a *severity parameter* from 1–5, corresponding to:

| Severity | σ (Noise Std) | Description |
|-----------|----------------|-------------|
| 1 | 0.05 | Very mild noise |
| 3 | 0.15 | Moderate scanner-like noise |
| 5 | 0.25 | Strong degradation simulating heavy domain shift |

This noise represents the **domain gap** found in real-world medical imaging — for example, differences between scanners, lighting, or magnification levels.

---

## Method
We used **CLIPArTT (CLIP Adaptive Robust Test-Time Tuning)** to perform *test-time adaptation* (TTA) on the PathMNIST dataset.

- **Base model:** ViT-B/32  
- **Top-K:** 3  
- **Tau (temperature):** 0.01  
- **Learning rate:** 0.0005  
- **Iterations per batch:** 15  
- **Batch size:** 128  

The TTA process updates only the **visual LayerNorm parameters** while keeping the rest of CLIP frozen.  
This allows the model to adapt efficiently to unseen, noisy data **without retraining**.

---

## Results
| Setting | Accuracy (%) |
|----------|---------------|
| **Zero-Shot CLIP** | 14.7 |
| **CLIPArTT (TTA)** | 19.4 |

The improvement shows that even though CLIP was never trained on medical data, adaptive tuning during inference helps the model better align with the corrupted medical domain.


---

## Key Insights
- CLIP’s visual encoder struggles with medical data because it was trained on general internet images.  
- Test-Time Adaptation (CLIPArTT) allows domain adaptation *without retraining* or labeled data.  
- Fine-tuning or retraining CLIP on medical images (e.g., with paired reports) could further improve accuracy.  
- The experiment highlights that **robust adaptation methods like CLIPArTT** can make foundation models more practical in critical medical applications.
"""


