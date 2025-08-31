<h2 align="center">
  <b>SCAR-3D: 3D Scene Change Modeling with Consistent Multi-View Aggregation</b>
  <br>
  <b><i>3DV 2026 Submission #332</i></b>
</h2>

<p align="center">
    <!-- Zirui Zhou, Junfeng Ni, Shujie Zhang, Yixin Chen, Siyuan Huang -->
    Anonymous Authors
</p>

<p align="center">
    <a href='#'>
        <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='#'>
        <img src='https://img.shields.io/badge/Dataset-CCS3D-blue?style=plastic&logo=Google%20sheets&logoColor=blue' alt='Dataset'>
    </a>
    <a href='#'>
        <img src='https://img.shields.io/badge/Code-Coming Soon-green?style=plastic&logo=github&logoColor=green' alt='Code'>
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

## 📦Installation

*(Code will be released upon acceptance.)*

```bash
# Coming soon
```

---

## Dataset

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

Download link and preprocessing scripts will be provided upon release.

---

## Usage

### Change Detection
```bash
# Example command (to be updated)
python detect_changes.py \
    --pre_images path/to/pre_change \
    --post_images path/to/post_change \
    --output path/to/masks
```

### Continual Reconstruction
```bash
# Example command (to be updated)
python reconstruct.py \
    --pre_model path/to/pre_change_gs \
    --post_images path/to/post_change \
    --change_masks path/to/masks \
    --output path/to/post_change_gs
```

---

## Results

### Quantitative Comparison (CCS3D)

| Method       | F1 ↑     | IoU ↑     |
|--------------|----------|-----------|
| Pixel-Diff   | 0.318    | 0.198     |
| Feature-Diff | 0.450    | 0.343     |
| 3DGS-CD      | 0.512    | 0.416     |
| **Ours**     | **0.724**| **0.650** |

### Qualitative Examples

<p align="center">
    <img src="assets/results.png" width=80%>
</p>

---

## Citation

```bibtex
<!-- @inproceedings{anonymous2026scar3d,
  title={SCAR-3D: 3D Scene Change Modeling with Consistent Multi-View Aggregation},
  author={Anonymous},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2026}
} -->
```

---

## Acknowledgements

<!-- This work was supported by anonymous institutions. -->
 We thank the authors of 3DGS, EfficientSAM, COLMAP, and Blender for their open-source contributions.

<!-- --- -->
<!-- **Note**: This is a confidential submission to 3DV 2026. Code and data will be released upon acceptance. -->


