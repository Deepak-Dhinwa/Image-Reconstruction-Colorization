# Dual-Task Image Processing: Reconstruction and Colorization

One-page summary: open `poster.html` in a browser.
Start: `.\run.ps1`

A deep learning project for simultaneous image reconstruction and colorization using a shared encoder-decoder architecture with DINOv2 and LoRA.

## Overview

This project implements a dual-decoder neural network architecture that performs two tasks simultaneously:
- **Image Reconstruction**: Reconstructing masked patches in grayscale images
- **Image Colorization**: Predicting color (a*b* channels) from grayscale input

The model uses DINOv2 ViT-S/14 as a frozen encoder with LoRA (Low-Rank Adaptation) for efficient fine-tuning, followed by a shared decoder that branches into task-specific heads.

## Quick Start

Use the existing environment, `recon-color_env`. It already matches the saved weights (PyTorch 2.5.1 with CUDA). Do not create a new environment or upgrade it.

From this folder:

```bash
recon-color_env\Scripts\python.exe inference.py
```

That writes five comparison images into `results/`. Five images from an earlier run are already there, and running it again replaces `result_0.png` through `result_4.png`. Training images are in `coco_train2017/train2017` and validation images are in `coco_val2017/val2017`.

`model_BEST_smallgpu.pth` is the checkpoint to load. `checkpoint_LATEST_smallgpu.pth` is the last training checkpoint. `train_improve_color.py` would keep training from the best checkpoint and is not needed to run inference.

### Training

```bash
recon-color_env\Scripts\python.exe train.py
```

## Model Architecture

- **Encoder**: DINOv2 ViT-S/14 (pretrained, frozen with LoRA adaptation)
- **Decoder**: Shared convolutional decoder with dual branches
- **Tasks**: RGB reconstruction + LAB colorization

## Results

The model was trained for 2 epochs on COCO dataset due to computational constraints. Reconstruction shows promising results, while colorization requires further training and optimization.

For detailed architecture, training methodology, and analysis, see [technical_report.md](technical_report.md).

