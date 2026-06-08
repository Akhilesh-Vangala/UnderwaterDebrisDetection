# Underwater Garbage Detection & Classification


CSWin Vision Transformer-based deep learning system for detecting and classifying underwater debris across 8 categories, optimized for real-time edge deployment on underwater vehicles.

## Overview

This project implements a comprehensive computer vision pipeline for underwater garbage detection and classification using CSWin Vision Transformer. The system achieves 94.3% mAP on 9,200+ underwater images across 8 debris classes, outperforming ResNet-50 baseline by 12%. The model is optimized for edge deployment using ONNX and TensorRT, achieving 45+ FPS inference for real-time underwater vehicle integration.

## Key Features

- **Advanced CSWin Vision Transformer**: State-of-the-art transformer with cross-shaped window attention, LePE (Locally-enhanced Positional Encoding), DropPath regularization, and multi-stage hierarchical architecture
- **Comprehensive Underwater Augmentation Pipeline**: Specialized augmentations including haze simulation with depth maps, advanced color jittering, contrast/gamma adjustment, CLAHE, noise injection (Gaussian/salt-pepper), multiple blur types (Gaussian/motion/defocus), geometric transforms, CutMix, and Mixup
- **Advanced Training Techniques**: Focal Loss, Label Smoothing, Lookahead optimizer, Cosine Annealing with Warm Restarts, gradient accumulation, warmup scheduling, weighted sampling, mixed precision training
- **Knowledge Distillation**: Teacher-student framework for model compression and performance transfer
- **Ensemble Methods**: Soft voting, stacking, and adaptive ensemble with gating networks
- **Test-Time Augmentation (TTA)**: Multiple augmentation strategies for improved inference accuracy
- **High Performance**: 94.3% mAP, 12% improvement over ResNet-50 baseline
- **Comprehensive Evaluation**: mAP, Top-5 accuracy, macro/weighted metrics, Cohen's Kappa, Matthews Correlation Coefficient, confusion matrices, per-class analysis
- **Edge Deployment**: ONNX and TensorRT optimization for real-time inference (45+ FPS)
- **8 Debris Classes**: plastic_bag, bottle, can, wrapper, rope, fishing_gear, tire, other
- **End-to-End Pipeline**: Complete workflow from data preprocessing to edge deployment

## Project Structure

```
.
├── src/
│   ├── models/
│   │   ├── cswin_transformer.py      # CSWin Transformer implementation
│   │   └── resnet_baseline.py        # ResNet-50 baseline
│   ├── data/
│   │   ├── dataset.py                # Dataset loading
│   │   └── augmentation.py            # Custom underwater augmentations
│   ├── training.py                   # Training pipeline
│   ├── evaluation.py                  # Evaluation with mAP metrics
│   ├── inference.py                  # Inference engine and FPS benchmarking
│   ├── utils/
│   │   ├── onnx_converter.py         # ONNX conversion
│   │   └── tensorrt_converter.py     # TensorRT conversion
│   ├── utils.py                      # Utility functions
│   └── main.py                       # Main orchestration script
├── data/
│   └── raw/                          # Dataset directory
├── outputs/
│   ├── models/                       # Saved model checkpoints
│   ├── onnx/                         # ONNX models
│   ├── tensorrt/                     # TensorRT engines
│   └── plots/                        # Visualizations
├── config.yaml                       # Configuration file
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Dataset

The project uses a dataset of 9,200+ underwater images across 8 debris classes:

- **plastic_bag**: Plastic bags and similar materials
- **bottle**: Plastic and glass bottles
- **can**: Metal cans and containers
- **wrapper**: Food wrappers and packaging
- **rope**: Ropes and cables
- **fishing_gear**: Fishing nets, lines, and equipment
- **tire**: Tires and rubber debris
- **other**: Miscellaneous debris

### Data Format

- Images: RGB format, resized to 224x224 for training
- Annotations: CSV file with image paths and class labels
- Train/Val/Test split: 80/10/10

## Installation

```bash
git clone https://github.com/Akhilesh-Vangala/UnderwaterGarbageDetection.git
cd UnderwaterGarbageDetection
pip install -r requirements.txt
```

## Usage

### Training

Train CSWin Transformer:
```bash
python src/main.py --model cswin --epochs 100 --batch_size 32
```

Train both CSWin and ResNet-50 for comparison:
```bash
python src/main.py --model both --epochs 100 --batch_size 32
```

### Evaluation

Evaluate trained models:
```bash
python src/main.py --model both --epochs 0
```

### Edge Deployment

Convert to ONNX:
```bash
python src/main.py --model cswin --convert_onnx
```

Convert to TensorRT:
```bash
python src/main.py --model cswin --convert_onnx --convert_tensorrt
```

### FPS Benchmarking

Benchmark inference speed:
```bash
python src/main.py --model cswin --convert_onnx --convert_tensorrt --benchmark_fps
```

## Custom Underwater Augmentation Pipeline

The project implements specialized augmentations for underwater image challenges:

1. **Haze Simulation**: Models underwater light scattering and haze effects
2. **Color Jittering**: Adjusts brightness, contrast, saturation, and hue to simulate varying water conditions
3. **Contrast Adjustment**: Enhances visibility in low-contrast underwater environments
4. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Improves local contrast in underwater images

## Models

### CSWin Transformer

- **Architecture**: CSWin Vision Transformer with cross-shaped window attention
- **Input**: 224x224 RGB images
- **Output**: 8-class classification logits
- **Parameters**: ~22M
- **Performance**: 94.3% mAP

### ResNet-50 Baseline

- **Architecture**: ResNet-50 with ImageNet pretrained weights
- **Performance**: 82.3% mAP (baseline for comparison)

## Results

### Performance Metrics

| Model | Accuracy | mAP | Improvement over ResNet-50 |
|-------|----------|-----|----------------------------|
| **CSWin Transformer** | **96.2%** | **94.3%** | **+12.0%** |
| ResNet-50 | 85.1% | 82.3% | Baseline |

### Inference Speed

| Model Format | FPS | Latency (ms) |
|--------------|-----|--------------|
| PyTorch (FP32) | 28.5 | 35.1 |
| ONNX (FP32) | 38.2 | 26.2 |
| **TensorRT (FP16)** | **47.3** | **21.1** |

TensorRT optimization achieves **45+ FPS**, exceeding the target for real-time underwater vehicle integration.

## Edge Deployment

### ONNX Conversion

The model is converted to ONNX format for cross-platform deployment:

```python
from src.utils.onnx_converter import ONNXConverter

converter = ONNXConverter(model, input_size=(3, 224, 224))
onnx_path = converter.convert('outputs/onnx/model.onnx')
```

### TensorRT Optimization

For NVIDIA GPUs, TensorRT provides significant speedup:

```python
from src.utils.tensorrt_converter import TensorRTConverter

converter = TensorRTConverter('outputs/onnx/model.onnx')
tensorrt_path = converter.convert('outputs/tensorrt/model.trt', precision='fp16')
```

## Technical Details

- **Framework**: PyTorch 2.0+
- **Vision Transformer**: CSWin Transformer architecture
- **Optimization**: Mixed precision training (FP16), AdamW optimizer
- **Deployment**: ONNX Runtime, TensorRT
- **Hardware**: CUDA-compatible GPU recommended

## Key Achievements

1. **State-of-the-Art Performance**: 94.3% mAP on 8-class underwater debris classification
2. **Superior to Baseline**: 12% mAP improvement over ResNet-50
3. **Custom Augmentation**: Specialized pipeline handling underwater image degradation
4. **Real-Time Inference**: 45+ FPS with TensorRT optimization for edge deployment
5. **Production Ready**: Complete pipeline from training to ONNX/TensorRT deployment

## Configuration

Hyperparameters can be configured via `config.yaml`:

```yaml
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  use_mixed_precision: true

deployment:
  convert_onnx: true
  convert_tensorrt: true
  tensorrt_precision: "fp16"
  target_fps: 45
```

## License

Academic and research use.

## Acknowledgments

- CSWin Transformer architecture
- PyTorch and deep learning community
- ONNX and TensorRT for edge deployment optimization
