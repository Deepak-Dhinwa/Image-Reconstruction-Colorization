import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from tqdm import tqdm
import numpy as np
import wandb
import time

from dataset import ReconColorDataset
from model import DinoLoRADualDecoder

# -------------------------
# Config (tweakable)
# -------------------------
TRAIN_DATA_DIR = "./coco_train2017/train2017"
VAL_DATA_DIR = "./coco_val2017/val2017"

# which pretrained checkpoint to start from (you trained 2 epochs earlier)
START_MODEL = "model_BEST_smallgpu.pth"     # preferred
ALT_CHECKPOINT = "checkpoint_LATEST_smallgpu.pth"

IMAGE_SIZE = 224
MASK_RATIO = 0.25

BATCH_SIZE = 1
ACCUMULATION_STEPS = 8
EPOCHS = 2                    # 1-2 epochs recommended
COLOR_LOSS_WEIGHT = 3.0       # strengthen color gradients

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0

SAVE_PATH = "model_COLORBOOST.pth"

USE_WANDB = True
WANDB_PROJECT = "recon-color-smallgpu"
WANDB_RUN_NAME = "color_only_finetune"

# -------------------------
# W&B init
# -------------------------
if USE_WANDB:
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config={
        "start_model": START_MODEL,
        "image_size": IMAGE_SIZE,
        "mask_ratio": MASK_RATIO,
        "batch_size": BATCH_SIZE,
        "accumulation_steps": ACCUMULATION_STEPS,
        "epochs": EPOCHS,
        "color_loss_weight": COLOR_LOSS_WEIGHT,
    })
    cfg = wandb.config
else:
    cfg = {"image_size": IMAGE_SIZE}

# -------------------------
# Datasets
# -------------------------
def get_loaders():
    train_ds = ReconColorDataset(TRAIN_DATA_DIR, image_size=IMAGE_SIZE, mask_ratio=MASK_RATIO)
    val_ds = ReconColorDataset(VAL_DATA_DIR, image_size=IMAGE_SIZE, mask_ratio=MASK_RATIO)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader

# -------------------------
# Model load & freeze rules
# -------------------------
def load_model_for_color_finetune(model_path=START_MODEL, alt_checkpoint=ALT_CHECKPOINT):
    model = DinoLoRADualDecoder(lora_rank=32).to(DEVICE)

    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=DEVICE)
    elif os.path.exists(alt_checkpoint):
        state = torch.load(alt_checkpoint, map_location=DEVICE)
    else:
        raise FileNotFoundError(f"Neither {model_path} nor {alt_checkpoint} found. Place your base model there.")

    # handle checkpoint vs plain state_dict
    if isinstance(state, dict) and ("model_state" in state or "model_state_dict" in state):
        sd = state.get("model_state", state.get("model_state_dict", state))
    else:
        sd = state

    model.load_state_dict(sd, strict=False)
    print("Loaded weights from", model_path if os.path.exists(model_path) else alt_checkpoint)

    # Freeze all parameters first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze only color branch and LoRA parameters
    unfreeze_count = 0
    for name, p in model.named_parameters():
        lname = name.lower()
        if "color_branch" in lname or "lora" in lname:
            p.requires_grad = True
            unfreeze_count += p.numel()

    print(f"Unfrozen params (color+LoRA): {unfreeze_count:,} parameters")
    # show trainable param count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,}")

    return model

# -------------------------
# Training loop
# -------------------------
def train_one_epoch(model, loader, optimizer, l1_loss, scaler, epoch):
    model.train()
    loop = tqdm(loader, desc=f"TrainColor E{epoch}", leave=False)
    total = 0.0
    use_cuda = DEVICE.startswith("cuda")
    autocast_ctx = torch.cuda.amp.autocast if use_cuda else torch.cpu.amp.autocast

    last_i = -1
    for i, (x, gt_rgb, gt_ab) in enumerate(loop):
        last_i = i
        x = x.to(DEVICE, non_blocking=True)
        gt_ab = gt_ab.to(DEVICE, non_blocking=True)

        torch.cuda.empty_cache()

        with autocast_ctx():
            _, out_color = model(x)
            out_color_f = out_color.float()
            loss_color = l1_loss(out_color_f, gt_ab) * COLOR_LOSS_WEIGHT
            loss_scaled = loss_color / ACCUMULATION_STEPS

        scaler.scale(loss_scaled).backward()

        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total += loss_color.item()
        loop.set_postfix(color_loss=loss_color.item())

        if USE_WANDB:
            wandb.log({
                "batch/color_loss": loss_color.item()
            })

    # flush remaining grads
    if last_i >= 0 and (last_i + 1) % ACCUMULATION_STEPS != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, l1_loss):
    model.eval()
    total = 0.0
    use_cuda = DEVICE.startswith("cuda")
    autocast_ctx = torch.cuda.amp.autocast if use_cuda else torch.cpu.amp.autocast
    for x, _, gt_ab in loader:
        x = x.to(DEVICE)
        gt_ab = gt_ab.to(DEVICE)
        with autocast_ctx():
            _, out_color = model(x)
            loss = l1_loss(out_color.float(), gt_ab) * COLOR_LOSS_WEIGHT
        total += loss.item()
    return total / max(1, len(loader))

# -------------------------
# Main
# -------------------------
def main():
    train_loader, val_loader = get_loaders()
    model = load_model_for_color_finetune()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=1e-3, weight_decay=1e-6)

    l1_loss = nn.L1Loss().to(DEVICE)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.startswith("cuda"))

    best_val = float("inf")
    for ep in range(EPOCHS):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, l1_loss, scaler, ep)
        val_loss = validate(model, val_loader, l1_loss)
        dt = time.time() - t0

        print(f"[EP {ep}] train_color_loss: {train_loss:.4f} | val_color_loss: {val_loss:.4f} | time {dt:.1f}s")
        if USE_WANDB:
            wandb.log({"epoch": ep, "train_color_loss": train_loss, "val_color_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            print("Saving model_COLORBOOST:", SAVE_PATH)
            torch.save(model.state_dict(), SAVE_PATH)

    print("Done. Best val:", best_val)
    if USE_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()
