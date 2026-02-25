# TornadoNet

**Real-Time Building Damage Detection with Ordinal Supervision**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](YOUR_ARXIV_LINK)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/crumeike/tornadonet-datasets)
[![Models](https://img.shields.io/badge/🤗-Models-blue)](https://huggingface.co/crumeike/tornadonet-checkpoints)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

TornadoNet is a comprehensive benchmark for automated street-level building damage assessment following tornado events. We systematically compare modern CNN-based (YOLO) and transformer-based (RT-DETR) object detection architectures and introduce ordinal-aware supervision strategies for multi-level damage classification.

<img width="921" height="1233" alt="image" src="https://github.com/user-attachments/assets/e20fcd5f-2857-44f5-8907-aa770c483962" />



### Key Features

✅ **3,333 street-view images** from 2021 Midwest tornado outbreak  
✅ **8,890 building instances** annotated with 5-level damage states (DS0-DS4)  
✅ **8 baseline models** across YOLO and RT-DETR architectures  
✅ **Ordinal-aware supervision** improving damage severity estimation by 4.8% mAP  
✅ **Real-time inference** up to 276 FPS on A100 GPUs

---

## Results

### Baseline Performance

| Model | mAP@0.5 (%) | F1 Score (%) | Ordinal Top-1 (%) | MAOE | FPS | Params (M) |
|-------|-------------|--------------|-------------------|------|-----|------------|
| YOLO11x | **46.05** | **49.40** | 85.20 | 0.76 | 66 | 56.8 |
| YOLOv8n | 40.98 | 45.11 | 84.01 | 0.78 | **276** | **3.0** |
| RT-DETR-L | 39.87 | 44.77 | **88.13** | **0.65** | 78 | 32.0 |

### Ordinal Supervision Impact

| Model | Configuration | mAP@0.5 (%) | Δ vs Baseline | Ordinal Top-1 (%) | MAOE |
|-------|---------------|-------------|---------------|-------------------|------|
| RT-DETR-L | Baseline | 39.87 | — | 88.13 | 0.65 |
| RT-DETR-L | ψ=0.5, K=1 | **44.70** | **+4.8** | **91.15** | **0.56** |
| RT-DETR-L | λ=0.05 | 43.36 | +3.5 | 89.54 | 0.61 |

---

## Installation

```bash
# Clone repository
git clone https://github.com/crumeike/TornadoNet.git
cd TornadoNet

# Create environment
conda create -n tornadonet python=3.10
conda activate tornadonet

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Download Dataset

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="crumeike/tornadonet-datasets",
    repo_type="dataset",
    local_dir="./data"
)
```

### Download Models

```python
from huggingface_hub import hf_hub_download

# Download best baseline
model_path = hf_hub_download(
    repo_id="crumeike/tornadonet-checkpoints",
    filename="tornadonet-yolo11-x-baseline/weights/best.pt"
)

# Download best ordinal model
ordinal_model = hf_hub_download(
    repo_id="crumeike/tornadonet-checkpoints",
    filename="tornadonet-rtdetr-l-ordinal-psi0.5-k1/weights/best.pt"
)
```

### Inference

```python
from ultralytics import YOLO

# Load model
model = YOLO(model_path)

# Run inference
results = model.predict(
    source="path/to/image.jpg",
    imgsz=896,
    conf=0.25
)

# Visualize
results[0].show()
```

### Training

```bash
# Train baseline model
python train.py \
    --model yolov8l \
    --data data/tornadonet.yaml \
    --epochs 250 \
    --imgsz 896 \
    --batch 16

# Train with ordinal supervision
python train.py \
    --model rtdetr-l \
    --data data/tornadonet.yaml \
    --epochs 250 \
    --imgsz 896 \
    --batch 16 \
    --ordinal \
    --psi 0.5 \
    --k 1
```

---

## Dataset Structure

```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── tornadonet.yaml
```

### Damage Classification

| Class | Label | Description | Count (Train/Val/Test) |
|-------|-------|-------------|------------------------|
| DS0 | Undamaged | No visible damage | 2,847 / 581 / 588 |
| DS1 | Slight | Minor roof/window damage | 1,789 / 372 / 369 |
| DS2 | Moderate | Significant roof damage | 951 / 219 / 212 |
| DS3 | Extensive | Major structural damage | 464 / 126 / 137 |
| DS4 | Complete | Total collapse | 133 / 44 / 58 |

---

## Available Models

### Baseline Models (8)

- `tornadonet-yolov8-{n/l/x}-baseline`
- `tornadonet-yolo11-{n/l/x}-baseline`
- `tornadonet-rtdetr-{l/x}-baseline`

### Ordinal Variants (2)

- `tornadonet-rtdetr-l-ordinal-psi0.5-k1` (Main contribution)
- `tornadonet-rtdetr-l-ordinal-lambda0.05` (Alternative approach)

All model checkpoints and datasets are available at:
- [🤗 tornadonet-checkpoints](https://huggingface.co/crumeike/tornadonet-checkpoints)
- [🤗 tornadonet-datasets](https://huggingface.co/crumeike/tornadonet-datasets)

## Citation

If you use TornadoNet in your research, please cite:

```bibtex
@article{umeike2025tornadonet,
  title={TornadoNet: Real-Time Building Damage Detection with Ordinal Supervision},
  author={Umeike, Robinson and Pham, Cuong and Hausen, Ryan and Dao, Thang and Crawford, Shane and Brown-Giammanco, Tanya and Lemson, Gerard and van de Lindt, John and Johnston, Blythe and Mitschang, Arik and Do, Trung},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## Acknowledgments

This work was supported by the Center for Risk-Based Community Resilience Planning (NIST Cooperative Agreement 70NANB15H044) and SciServer (NSF Award ACI-1261715).

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Contact

**Robinson Umeike** - The University of Alabama  
📧 crumeike@crimson.ua.edu  
🔗 [GitHub](https://github.com/crumeike)

For questions or collaborations, please open an issue or contact the authors.
