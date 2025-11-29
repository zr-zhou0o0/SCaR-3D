<h2 align="center">
  <b>SCaR-3D: 3D Scene Change Modeling with Consistent Multi-View Aggregation</b>
  <br>
  <b><i>3DV 2026</i></b>
</h2>

<p align="center">
    <a href="https://github.com/zr-zhou0o0">Zirui Zhou</a><sup>1</sup>,
    <a href="https://dali-jack.github.io/Junfeng-Ni/">Junfeng Ni</a><sup>1,2</sup>,
    <a href="https://hishujie.github.io/">Shujie Zhang</a><sup>1</sup>,
    <a href="https://yixchen.github.io/">Yixin Chen</a><sup>2✉</sup>,
    <a href="https://siyuanhuang.com/">Siyuan Huang</a><sup>2✉</sup>
</p>

<p align="center">
    <sup>1</sup>Tsinghua University &nbsp;&nbsp; <sup>2</sup>State Key Laboratory of General Artificial Intelligence, BIGAI
</p>

<p align="center">
    <a href='https://zr-zhou0o0.github.io/SCaR-3D.github.io/'>
        <img src='https://img.shields.io/badge/Project-Page-Green?style=plastic&logo=googlechrome&logoColor=green' alt='Project Page'>
    </a>
    <a href='https://zr-zhou0o0.github.io/SCaR-3D.github.io/'>
        <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='#'>
        <img src='https://img.shields.io/badge/Dataset-CCS3D-blue?style=plastic&logo=Google%20sheets&logoColor=blue' alt='Dataset'>
    </a>
    <a href='https://github.com/zr-zhou0o0/SCaR-3D'>
        <img src='https://img.shields.io/badge/Code-Github-black?style=plastic&logo=github&logoColor=white' alt='Code'>
    </a>
</p>

<p align="center">
    <img src="assets/teaser.png" width=90%>
</p>

SCAR-3D is a novel 3D scene change detection and reconstruction framework that identifies object-level changes from dense pre-change and sparse post-view images. It leverages a signed-distance-based 2D differencing module, multi-view aggregation with voting and pruning, and segmentation validation to produce accurate and consistent 3D change masks. The method also supports continual scene reconstruction by selectively updating dynamic regions.

---

## 🌟Features

- **Multi-view Consistent Change Detection**: Aggregates 2D differences into a unified 3D representation with voting and pruning.
- **Signed-Distance-Based Localization**: Captures directional changes in feature space for robust detection.
- **Segmentation Validation**: Uses EfficientSAM to refine change masks and improve accuracy.
- **Continual Reconstruction**: Updates only changed regions to maintain scene consistency and reduce artifacts.
- **Synthetic Dataset (CCS3D)**: Provides editable indoor scenes with diverse change types for controlled evaluation.

---

## 🔨 Installation


#### Prerequisites
- CUDA 12.4 or higher
- Python 3.8
- Conda (recommended)

#### Step 1: Clone the Repository
```bash
git clone https://github.com/zr-zhou0o0/SCaR-3D.git
cd SCaR-3D
```

#### Step 2: Create Conda Environment
```bash
conda create -n scar3d python=3.8 -y
conda activate scar3d
```

#### Step 3: Install PyTorch and CUDA Dependencies
```bash
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### Step 4: Install Other Dependencies
```bash
pip install -r requirements.txt
```

#### Step 5: Install Other Required Packages from Source
```bash
pip install git+https://github.com/yformer/EfficientSAM.git@c9408a74b1db85e7831977c66e9462c6f4891729

pip install git+https://github.com/camenduru/simple-knn.git

pip install git+https://github.com/rahul-goel/fused-ssim.git

pip install submodules/diff-point-rasterization
```

## 📦 Pretrained Weights

---

## 📊 Dataset

### CCS3D (Controllable Change in 3D Scenes)
A synthetic dataset built with Blender, featuring four complex indoor scenes:
- `Desk`
- `Bookcase`
- `Livingroom`
- `Bedroom`


Each scene supports:
- Multiple change types: insertion, removal, translation, rotation, mixed.
- Complex camera trajectories simulating real-world navigation.
- Fine-grained object-level annotations.

Download the CCS3D dataset [here](#).

### 3DGS-CD 
Our processed 3DGS-CD dataset can be downloaded [here](#).

### Customized Datasets
<!-- #### Dataset Creation -->
#### Dataset Structure

The dataset should be organized as follows:

```
dataset/
└── <dataset_name>/
    ├── <scene_name>/
    │   ├── images/          # All images
    │   ├── train-pre/       # Pre-change training images
    │   ├── train-post/      # Post-change training images
    │   ├── test-pre/        # Pre-change test images
    │   ├── test-post/       # Post-change test images
    │   ├── gt-pre-mask/     # Ground truth masks for pre-change
    │   ├── gt-post-mask/    # Ground truth masks for post-change
    │   ├── sparse/          # COLMAP sparse reconstruction
    │   └── <scene_name>.db  # COLMAP database
    └── ...
```

```bash
# Coming Soon
```

---

## 💡Usage

### Change Detection
```bash
# CCS3D Dataset
bash run_ccs3d.sh

# 3DGS-CD Dataset
bash run_3dgs_cd.sh

# If you want to train 3DGS models from scratch, use the following command:
bash run_train.sh
```

<!-- ### Continual Reconstruction
```bash
# Coming Soon
``` -->

---

## Results

## Quantitative Change Detection Results

### Results on CCS3D Dataset

| Method | Livingroom F1 | Livingroom IoU | Desk F1 | Desk IoU | Bookcase F1 | Bookcase IoU | Bedroom F1 | Bedroom IoU | Average F1 | Average IoU |
|--------|---------------|----------------|---------|----------|-------------|--------------|------------|-------------|------------|-------------|
| Pixel-Diff | 0.273 | 0.162 | 0.398 | 0.254 | 0.315 | 0.201 | 0.286 | 0.176 | 0.318 | 0.198 |
| Feature-Diff | 0.420 | 0.302 | 0.480 | 0.323 | 0.320 | 0.256 | 0.705 | 0.584 | 0.450 | 0.343 |
| CL-Splat | 0.789 | 0.657 | 0.567 | 0.399 | 0.294 | 0.199 | 0.501 | 0.341 | 0.538 | 0.399 |
| MV3DCD | 0.478 | 0.329 | 0.291 | 0.178 | 0.449 | 0.295 | 0.547 | 0.413 | 0.441 | 0.304 |
| 3DGS-CD | 0.897 | 0.815 | 0.525 | 0.408 | 0.477 | 0.353 | 0.148 | 0.089 | 0.512 | 0.416 |
| **Ours** | **0.955** | **0.914** | **0.610** | **0.477** | **0.423** | **0.377** | **0.909** | **0.834** | **0.724** | **0.650** |


### Qualitative Examples

<p align="center">
    <img src="assets/results.png" width=80%>
</p>

---

## Citation

Comming soon

<!-- ```bibtex
@inproceedings{anonymous2026scar3d,
  title={SCAR-3D: 3D Scene Change Modeling with Consistent Multi-View Aggregation},
  author={Anonymous},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2026}
}
``` -->

---



