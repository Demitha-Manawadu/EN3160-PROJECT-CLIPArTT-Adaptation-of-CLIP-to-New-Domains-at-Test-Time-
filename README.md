# CLIPArTT – Adaptation of CLIP to New Domains at Test Time (WACV 2025 Reimplementation + Medical Domain Extension)

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

| Dataset | Metric | Zero-shot CLIP | CLIPArTT (ours) | Δ Improvement |
|----------|--------|----------------|------------------|---------------|
| CIFAR-10 (clean) | Top-1 Acc (%) | 88.8 | **88.9** | +0.1 |
| CIFAR-10-C (Gaussian Noise, s=5) | Top-1 Acc (%) | 35.6 | **36.5** | +0.9 |
| CIFAR-100-C (avg over 15 corr.) | Top-1 Acc (%) | 29.4 | **41.5** | +12.1 |

> ⚙️ Results reproduced using `ViT-B/32` and batch size = 128.  
> Accuracy improvements are consistent across most corruption types (Gaussian, Defocus, Glass, etc.).

---

## Extension to Medical Images

To study CLIP’s robustness in **medical imaging**, we extended CLIPArTT to handle **domain-specific data** and **grayscale inputs**.

### Datasets
| Dataset | Description |
|----------|--------------|
| **CheXpert (clean)** | High-quality chest X-rays used as a “clean” domain. |
| **CheXphoto (corrupted)** | Real-world, mobile-captured photos of printed X-rays, simulating domain shift. |
| **Synthetic corruption** | Optionally added Gaussian noise, motion blur, JPEG compression, etc., for testing CLIP’s invariance. |
