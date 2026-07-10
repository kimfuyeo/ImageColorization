# ImageColorization: Cascaded Architecture for Historical Media

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C.svg)](https://pytorch.org/)
[![CUDA 13.2](https://img.shields.io/badge/CUDA-13.2-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract
The colorization of historical grayscale media is a complex inverse problem characterized by high semantic ambiguity and physical archival degradation. Fully automated regression models inherently suffer from the Reference Paradox and a pervasive sepia bias, hallucinating mathematically safe but historically inaccurate brownish tones to minimize $L_2$ spatial error.

This repository contains the official implementation of a novel two-stage cascaded deep learning architecture designed to eradicate this bias. By fundamentally decoupling high-frequency geometric reconstruction (via a DDColor baseline) from localized semantic coloration (via a UniColor interactive transformer), this pipeline establishes a highly controlled, enterprise-ready restoration tool capable of restoring historically accurate chrominance to both static and cinematic legacy media.

## Key Contributions
1. **Hybrid Cascaded Architecture:** A dual-stage pipeline uniting autonomous macroscopic feature extraction with unified multimodal exemplar refinement.
2. **Accurate Colorization:** Deterministic mathematical suppression of network desaturation, achieving a 51.46% reduction in chromatic error ($\Delta E$).
3. **Temporal Cinematic Synthesis:** A robust pipeline utilizing patch-based optical flow (EBSynth) to lock semantic color tokens to moving geometry, neutralizing inter-frame temporal artifacting.

## Dataset Access
The quantitative benchmarking subset (ImageNet) and the curated qualitative historical media (Malaysian archival photography) used to evaluate this architecture are hosted externally due to size constraints. 

* **[Download the Evaluation Dataset (Google Drive)](https://drive.google.com/drive/folders/1EH7et9uIXpYH7nWeGAXlqcwHctb8PLj8?usp=sharing)**

Once downloaded, extract the contents into the root directory of the repository (e.g., into the `./input_data/` and `./Analysis/` folders) to seamlessly replicate the batch inference and domain validation pipelines.

* **ImageNet Dataset:** J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A large-scale hierarchical image database," in *2009 IEEE Conference on Computer Vision and Pattern Recognition*, 2009, pp. 248–255.
  
* **Old Malaysia Photos:** Nandini_Bala, “Ancient photos reveal how KL used to look like 50 years ago!,” SAYS, Oct. 26, 2016. [Online]. Available: https://says.com/my/lifestyle/old-photos-of-kl
---

## Repository Structure

```text
├── Baseline/               # Standalone control architectures
│   ├── BigColor/
│   ├── Colorformer/
│   ├── DDColor/
│   ├── DeOldify/           # Includes PyTorch 2.6 security bypass wrapper
│   └── UniColor/
├── Multimodal/             # The proposed cascaded pipeline
│   ├── app.py              # Gradio UI for end-to-end interactive synthesis
│   ├── batch_executor.py   # High-throughput batch inference script
│   └── src/                # Wrapper scripts linking DDColor and UniColor
├── Video/                  # Temporal synthesis pipeline
│   ├── 01_extract_keyframes.py
│   ├── 02_colorize_keyframes.py
│   └── 03_ebsynth_propagation.py
├── Analysis/               # Evaluation footprint and dataset validation
│   ├── Baseline/           # Baseline metrics
│   ├── metrics/            # Raw .csv and .json data (PSNR, LPIPS, UIC, etc.)
│   └── gallery/            # Visual ablation grids and boundary condition examples
├── Results/                # Final output imagery and comparisons
└── README.md
