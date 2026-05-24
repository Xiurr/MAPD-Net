# MAPD-Net: Modality-Aware Anatomy-Pathology Disentanglement for Incomplete Multi-Modality Brain Tumor Segmentation

<div align="center">
<p align="center">
    <img src="figures/fig2.png" width="1000" alt="MAPD-Net Architecture"/> <br />
</p>
</div>

## Overview

**MAPD-Net** is a **Modality-aware Anatomy-Pathology Disentanglement Network** for incomplete multi-modality brain tumor segmentation. Multimodal MRI provides complementary anatomical and pathological information, but missing modalities frequently occur in clinical practice and challenge the stability and generalization of segmentation models.

Existing methods typically rely on modality synthesis, knowledge distillation, or shared representation learning. However, synthesis-based methods may introduce artifacts, distillation-based methods often require separate models for different missing-modality combinations, and shared representation learning may cause semantic entanglement between modality-invariant anatomy and modality-specific pathology.

MAPD-Net addresses this issue by explicitly separating **modality-shared anatomical representations** from **modality-exclusive pathological representations** at the representation level, avoiding pixel-level image reconstruction while preserving discriminative pathological cues.

MAPD-Net consists of three synergistic components:

1. **HPE: Heterogeneous Pathology Encoder**  
   Uses modality-tailored branches to capture distinctive pathological responses from T1, T1ce, T2, and FLAIR images.

2. **APDM: Anatomy-Pathology Disentanglement Module**  
   Promotes explicit feature separation through dual-path adversarial learning and a multi-component disentanglement loss.

3. **PCFM: Pathology-Conditioned Fusion Module**  
   Re-injects pathological priors into the modality-shared anatomical prototype through channel modulation and spatial routing for final segmentation.

---

## Code Availability & Release Plan

The released version focuses on the proposed architecture and the key anatomy-pathology disentanglement design, including:

- The proposed MAPD-Net architecture and core model components (`modeling/`)
- The implementation of the key modules: **HPE**, **APDM**, and **PCFM**
- Multi-component disentanglement loss functions, including anatomical consistency loss, pathological uniqueness loss, and adversarial semantic constraint loss (`utils/loss.py`)
- Basic quantitative metrics and evaluation utilities (`utils/metrics.py`)
- A lightweight forward verification example with random inputs

Complete training configurations, BraTS preprocessing scripts, full dataloader implementation and evaluation pipelines will be released upon paper acceptance.

---

## Content

- [Experimental Results](#experimental-results)
- [Project Structure](#project-structure-core-implementation)
- [Method Components](#method-components)
- [Environment and Dataset Information](#environment-and-dataset-information)
- [Forward Verification](#forward-verification-core-release)
- [Citation](#citation)

---

## Experimental Results

To comprehensively evaluate the effectiveness and robustness of MAPD-Net for multimodal brain tumor segmentation, we conducted extensive experiments on the **BraTS 2018** and **BraTS 2020** datasets, especially under complex missing-modality scenarios. Following the incomplete multi-modality setting, experiments were performed across **15 possible modality combinations** constructed from four MRI modalities: T1, T1ce, T2, and FLAIR.

The segmentation performance is reported using the Dice coefficient on three clinically significant tumor regions:

- **WT**: Whole Tumor
- **TC**: Tumor Core
- **ET**: Enhancing Tumor

Experimental results demonstrate that MAPD-Net achieves superior segmentation performance compared with state-of-the-art methods in various missing-modality scenarios. The results validate the effectiveness of explicit modality-aware anatomy-pathology disentanglement and show the robustness of MAPD-Net when modalities are missing.

### Overall Dice Performance

| Dataset | WT | TC | ET | Avg. |
|---|---:|---:|---:|---:|
| BraTS 2018 | 87.4 | 80.5 | 64.4 | 77.4 |
| BraTS 2020 | 89.0 | 82.5 | 65.2 | 78.9 |

The full results over all 15 missing-modality combinations are reported in the submitted manuscript.

### BraTS 2018 Results

<p align="center">
    <img src="figures/chart1.png" width="1000" alt="BraTS 2018 Results"/> <br />
</p>

### BraTS 2020 Results

<p align="center">
    <img src="figures/chart2.png" width="1000" alt="BraTS 2020 Results"/> <br />
</p>

### Qualitative Comparison

The qualitative comparison illustrates the segmentation performance of MAPD-Net under representative missing-modality scenarios. MAPD-Net produces segmentation results that are more spatially consistent with the ground truth across different modality combinations. In particular, it better preserves irregular ET boundaries and suppresses false positives when key modalities such as T1ce or FLAIR are unavailable.

<p align="center">
    <img src="figures/fig5.png" width="1000" alt="Qualitative Comparison"/> <br />
</p>

---

## Project Structure Core Implementation

```text
.
├── modeling/               # Proposed MAPD-Net architecture
│   ├── HPE.py              # Heterogeneous Pathology Encoder
│   └── ensemble/           # APDM and PCFM core logic
├── utils/                  # Essential utilities
│   ├── loss.py             # Disentanglement losses: L_cons, L_uni, and L_apdm
│   └── metrics.py          # Dice coefficient and basic evaluation metrics
├── figures/                # Architecture, quantitative results, and qualitative comparison
├── mypath.py               # Dataset path configuration template
├── train.py                # Lightweight forward verification entry
├── README.md               # Project documentation
└── requirements.txt        # Environment dependencies
```

---

## Method Components

### Heterogeneous Pathology Encoder

The Heterogeneous Pathology Encoder (HPE) employs four distinct, modality-tailored branches after a shared backbone network to extract modality-specific pathological features. Each branch is designed according to the imaging characteristics of the corresponding modality:

- The **T1 branch** leverages the sensitivity of T1 images to structural gradients and extracts high-frequency edge features.
- The **T1ce branch** captures irregular enhancing-region patterns using intensity gating and deformable convolution.
- The **T2 branch** models heterogeneous tumor-core-related patterns through a multi-path structure.
- The **FLAIR branch** is designed to model elongated and spatially diffuse edema-related patterns using strip convolutions and dilated convolution.

Through these differentiated extraction strategies, HPE preserves modality-exclusive pathological fingerprints that are crucial for robust segmentation under incomplete multi-modality inputs.

### Anatomy-Pathology Disentanglement Module

The Anatomy-Pathology Disentanglement Module (APDM) is designed to achieve explicit feature-level anatomy-pathology separation. It consists of two adversarial paths:

- The **Anatomical Adversarial Path** uses a Gradient Reversal Layer (GRL) to suppress modal attributes in anatomical features and encourage modality-invariant anatomical representations.
- The **Pathology Modulation Path** encourages modality-exclusive pathological features to retain modality-discriminative information and converts them into pathology modulation parameters.

Together with the multi-component disentanglement loss, APDM constrains anatomical features to be modality-independent while preserving modality-specific pathological semantics.

The disentanglement objective includes:

- Anatomical consistency loss
- Pathological uniqueness loss
- Adversarial semantic constraint loss

### Pathology-Conditioned Fusion Module

After disentanglement, the Pathology-Conditioned Fusion Module (PCFM) uses pathology modulation parameters as a conditioning guide to adaptively re-inject modality-specific pathological information into the modality-shared anatomical prototype.

PCFM performs:

- **Channel modulation**, which activates the discriminative power of each modality's pathological features.
- **Spatial routing**, which adaptively determines the effective spatial regions for injecting pathological information.

This pathology-conditioned fusion process maximizes the diagnostic utility of modality-exclusive pathological signals while preserving the integrity of anatomical structures.

---

## Environment and Dataset Information

### Environment

The recommended environment is:

- `Python 3.8+`
- `PyTorch 1.10+`
- `CUDA 11.3+`

Install the dependencies as follows:

```bash
# Clone the repository
git clone https://github.com/Xiurr/MAPD-Net.git
cd MAPD-Net

# Install dependencies
pip install -r requirements.txt
```

### Dataset Information

The experiments in the paper were conducted on the BraTS 2018 and BraTS 2020 datasets. These datasets contain preoperative multimodal MRI scans from multiple institutions and cover glioma patients with significant morphological and histological heterogeneity.

Each subject includes four MRI modalities:

- T1-weighted image: **T1**
- Contrast-enhanced T1-weighted image: **T1ce**
- T2-weighted image: **T2**
- Fluid-Attenuated Inversion Recovery image: **FLAIR**

Following the BraTS evaluation protocol, the original annotations are aggregated into three clinically significant regions:

- Enhancing Tumor: **ET**
- Tumor Core: **TC**
- Whole Tumor: **WT**

Raw BraTS datasets are not included in this repository due to dataset license restrictions. Please obtain the datasets from the official BraTS challenge data portals and comply with the corresponding data-use agreements.

The expected BraTS-style subject structure is:

```text
Subject_ID/
├── *_t1.nii.gz
├── *_t1ce.nii.gz
├── *_t2.nii.gz
├── *_flair.nii.gz
└── *_seg.nii.gz
```

---

## Forward Verification Core Release

This core release supports lightweight forward verification with random inputs. It is intended to verify the model logic rather than reproduce the full training and evaluation pipeline.

Run:

```bash
python train.py
```

### Missing-Modality Simulation

The modality index convention used in this repository is:

```text
0: T1
1: T1ce
2: T2
3: FLAIR
```

To simulate missing modalities, pass indices in `{0,1,2,3}`:

```bash
python train.py --missing "1,3"
```

The above example simulates the case where **T1ce** and **FLAIR** are missing.


## Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@article{zhang2026mapd,
  title={MAPD-Net: Modality-Aware Anatomy-Pathology Disentanglement for Incomplete Multi-Modality Brain Tumor Segmentation},
  author={Zhang, Ting and Li, Xiuhan and Liu, Zhaoying},
  journal={},
  year={2026}
}

