# TornadoNet

**Real-Time Building Damage Detection with Ordinal Supervision**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2603.11557)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/crumeike/tornadonet-datasets/tree/main)
[![Models](https://img.shields.io/badge/🤗-Models-blue)](https://huggingface.co/crumeike/tornadonet-checkpoints/tree/main)
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

## Installation

```bash
# Clone repository
git clone https://github.com/crumeike/TornadoNet.git
cd TornadoNet/main

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
    filename="tornadonet-yolo11-x-baseline/best.pt"
)

# Download best ordinal model
ordinal_model = hf_hub_download(
    repo_id="crumeike/tornadonet-checkpoints",
    filename="tornadonet-rtdetr-l-ordinal-psi0.5-k1/best.pt"
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

| Class | Label | Description |
|-------|-------|-------------|
| DS0 | Undamaged | No visible damage |
| DS1 | Slight | Minor roof/window damage |
| DS2 | Moderate | Significant roof damage |
| DS3 | Extensive | Major structural damage |
| DS4 | Complete | Total collapse |

**Dataset splits:** 6,184 train / 1,342 val / 1,364 test (75% / 15% / 15%)

---

## Results

### Baseline Performance*

| Model | Type | mAP@0.5 | F1 Score | Ordinal Top-1 | MAOE | FPS | Params | Download |
|-------|------|---------|----------|---------------|------|-----|--------|----------|
| YOLOv8-n | CNN | 40.98% | 45.11% | 84.01% | 0.78 | 276 | 3.0M | [📥](https://huggingface.co/crumeike/tornadonet-checkpoints/tree/main/tornadonet-yolov8-n-baseline) |
| YOLOv8-l | CNN | 42.09% | 46.41% | 84.19% | 0.78 | 91 | 43.6M | [📥](https://huggingface.co/crumeike/tornadonet-checkpoints/tree/main/tornadonet-yolov8-l-baseline) |
| YOLOv8-x | CNN | 41.84% | 46.24% | 83.04% | 0.81 | 68 | 68.1M | [📥](https://huggingface.co/crumeike/tornadonet-checkpoints/tree/main/tornadonet-yolov8-x-baseline) |
| YOLO11-n | CNN | 41.14% | 45.73% | 84.79% | 0.77 | 239 | 2.6M | [📥](https://huggingface.co/crumeike/tornadonet-checkpoints/tree/main/tornadonet-yolo11-n-baseline) |
| YOLO11-l | CNN | 40.44% | 44.41% | 83.75% | 0.79 | 96 | 25.3M | [📥](https://huggingface.co/crumeike/tornadonet-checkpoints/tree/main/tornadonet-yolo11-l-baseline) |
| YOLO11-x | CNN | **46.05%** | **49.40%** | 85.20% | 0.76 | 66 | 56.8M | [📥](https://huggingface.co/crumeike/tornadonet-checkpoints/tree/main/tornadonet-yolo11-x-baseline) |
| RT-DETR-L | Transformer | 39.87% | 44.77% | **88.13%** | **0.65** | 78 | 32.0M | [📥](https://huggingface.co/crumeike/tornadonet-checkpoints/tree/main/tornadonet-rtdetr-l-baseline) |
| RT-DETR-X | Transformer | 35.75% | 41.54% | 87.74% | 0.67 | 79 | 65.5M | [📥](https://huggingface.co/crumeike/tornadonet-checkpoints/tree/main/tornadonet-rtdetr-x-baseline) |

###  Ordinal Supervision Impact*

| Model | Configuration | mAP@0.5 | Δ vs Baseline | Ordinal Top-1 | MAOE | Download |
|-------|---------------|---------|---------------|---------------|------|----------|
| RT-DETR-L | Baseline | 39.87% | — | 88.13% | 0.65 | [📥](https://huggingface.co/crumeike/tornadonet-checkpoints/tree/main/tornadonet-rtdetr-l-baseline) |
| RT-DETR-L | ψ=0.5, K=1 | **44.70%** | **+4.8 pp** | **91.15%** | **0.56** | [📥](https://huggingface.co/crumeike/tornadonet-checkpoints/tree/main/tornadonet-rtdetr-l-ordinal-psi0.5-k1) |
| RT-DETR-L | λ=0.05 | 43.36% | +3.5 pp | 89.54% | 0.61 | [📥](https://huggingface.co/crumeike/tornadonet-checkpoints/tree/main/tornadonet-rtdetr-l-ordinal-lambda0.05) |

\* *Values represent mean ± std across 3 random seeds (paper). Downloaded checkpoints are from the best-performing seed.*

**Legend:**
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **Ordinal Top-1**: Percentage of predictions within ±1 damage class
- **MAOE**: Mean Absolute Ordinal Error (lower is better)
- **FPS**: Frames per second on NVIDIA A100 GPU
- **Δ**: Change compared to baseline (pp = percentage points)

## Available Models

### Baseline Models (8)

- `tornadonet-yolov8-{n/l/x}-baseline`
- `tornadonet-yolo11-{n/l/x}-baseline`
- `tornadonet-rtdetr-{l/x}-baseline`

### Ordinal Variants (2)

- `tornadonet-rtdetr-l-ordinal-psi0.5-k1` (Main contribution)
- `tornadonet-rtdetr-l-ordinal-lambda0.05` (Alternative approach)

All pre-trained model checkpoints and datasets are available at 🤗HuggingFace: 
- [Models](https://huggingface.co/crumeike/tornadonet-checkpoints/tree/main)
- [Datasets](https://huggingface.co/crumeike/tornadonet-checkpoints/tree/main)


---

## Citation

If you use TornadoNet in your research, please cite:

```bibtex
@article{umeike2026tornadonet,
  title={TornadoNet: Real-Time Building Damage Detection with Ordinal Supervision},
  author={Umeike, Robinson and Pham, Cuong and Hausen, Ryan and Dao, Thang and Crawford, Shane and Brown-Giammanco, Tanya and Lemson, Gerard and van de Lindt, John and Johnston, Blythe and Mitschang, Arik and Do, Trung},
  journal={arXiv preprint arXiv:2603.11557},
  year={2026}
}
```

---

## Acknowledgments

The authors acknowledge the Center for Risk-Based Community Resilience Planning, a NIST-funded Center of Excellence (Cooperative Agreement 70NANB15H044), for providing the street-view imagery dataset, and SciServer (sciserver.org), developed at Johns Hopkins University and funded by NSF Award ACI-1261715, for computational resources. These resources were instrumental in the development and validation of the models presented in this work.

---

## License

**Model and Code:** MIT License - see [LICENSE](LICENSE) file  
**Dataset:** CC BY 4.0 - see [DATA_LICENSE](DATA_LICENSE) file

---

## Contact

**Robinson Umeike** - The University of Alabama  
📧 crumeike@crimson.ua.edu  
🔗 [GitHub](https://github.com/crumeike)

For questions or collaborations, please open an issue or contact the authors.
