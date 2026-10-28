import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from tqdm import tqdm
import os
import numpy as np
import wandb
import time

from dataset import ReconColorDataset
from model import DinoLoRADualDecoder

# -------------------------
# Quick config for 4GB GPU
# -------------------------
TRAIN_DATA_DIR = "./coco_train2017/train2017"
VAL_DATA_DIR = "./coco_val2017/val2017"

IMAGE_SIZE = 224
MASK_RATIO = 0.25          # easier at start (helps quick learning)
LORA_RANK = 32

BATCH_SIZE = 1             # must be 1 for 4GB GPU
ACCUMULATION_STEPS = 8     # effective batch = 8 (1 * 8)
EPOCHS = 7                 # you said you can run ~6-7 epochs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_WORKERS = 0
SANITY_CHECK = False

BEST_MODEL_PATH = "model_BEST_smallgpu.pth"
LATEST_CHECKPOINT_PATH = "checkpoint_LATEST_smallgpu.pth"

# color loss weight multiplier (gives stronger color supervision)
COLOR_LOSS_WEIGHT = 2.0

# WandB (optional) - set WANDB_API_KEY env or run locally without W&B
USE_WANDB = True
WANDB_PROJECT = "Image-Reconstruction-and-Colorization"

# -------------------------
# W&B init
# -------------------------
if USE_WANDB:
    wandb.init(project=WANDB_PROJECT, config={
        "image_size": IMAGE_SIZE,
        "mask_ratio": MASK_RATIO,
        "lora_rank": LORA_RANK,
        "batch_size": BATCH_SIZE,
        "accumulation_steps": ACCUMULATION_STEPS,
        "epochs": EPOCHS,
        "color_loss_weight": COLOR_LOSS_WEIGHT,
    })
    cfg = wandb.config
else:
    cfg = {
        "image_size": IMAGE_SIZE,
        "mask_ratio": MASK_RATIO,
        "lora_rank": LORA_RANK,
        "batch_size": BATCH_SIZE,
        "accumulation_steps": ACCUMULATION_STEPS,
        "epochs": EPOCHS,
        "color_loss_weight": COLOR_LOSS_WEIGHT,
    }

# -------------------------
# Dataloaders
# -------------------------
def setup_dataloaders():
    print("Setting up datasets...")
    train_ds = ReconColorDataset(TRAIN_DATA_DIR, image_size=IMAGE_SIZE, mask_ratio=MASK_RATIO)
    val_ds = ReconColorDataset(VAL_DATA_DIR, image_size=IMAGE_SIZE, mask_ratio=MASK_RATIO)

    if SANITY_CHECK:
        n = min(32, len(train_ds))
        train_ds = Subset(train_ds, np.arange(n))
        val_ds = Subset(val_ds, np.arange(n))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader

# -------------------------
# Checkpoints
# -------------------------
def save_checkpoint(epoch, model, optimizer, best_val_loss, scaler):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(ckpt, LATEST_CHECKPOINT_PATH)

def load_checkpoint(model, optimizer, scaler):
    start_epoch = 0
    best_val_loss = float("inf")
    if os.path.exists(LATEST_CHECKPOINT_PATH) and not SANITY_CHECK:
        print(f"Loading checkpoint {LATEST_CHECKPOINT_PATH}")
        ckpt = torch.load(LATEST_CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"Resuming from epoch {start_epoch}")
    return start_epoch, best_val_loss

# -------------------------
# Model / optimizer setup
# -------------------------
def build_model_and_optimizer():
    model = DinoLoRADualDecoder(lora_rank=LORA_RANK)

    # Try enabling gradient checkpointing on encoder if available (saves memory at cost of time)
    try:
        # timm/vit sometimes supports gradient checkpointing via .set_grad_checkpointing() or .gradient_checkpointing_enable()
        if hasattr(model.encoder, "set_grad_checkpointing"):
            model.encoder.set_grad_checkpointing(True)
            print("Enabled encoder.set_grad_checkpointing(True)")
        elif hasattr(model.encoder, "gradient_checkpointing_enable"):
            model.encoder.gradient_checkpointing_enable()
            print("Enabled encoder.gradient_checkpointing_enable()")
    except Exception:
        pass

    model.to(DEVICE)

    # Parameter groups - small lr for encoder, larger for decoder, LoRA slightly larger
    encoder_params, decoder_params, lora_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if "lora" in lname:
            lora_params.append(p)
        elif lname.startswith("encoder"):
            encoder_params.append(p)
        else:
            decoder_params.append(p)

    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": 1e-5})
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": 5e-5})  # slightly reduced lr for stability
    if lora_params:
        param_groups.append({"params": lora_params, "lr": 1e-3})

    optimizer = optim.AdamW(param_groups, weight_decay=1e-6)
    return model, optimizer

# -------------------------
# Training / Validation loops
# -------------------------
def train_one_epoch(model, loader, optimizer, l1_loss, ssim_loss, scaler, epoch, log_images_every=2):
    model.train()
    loop = tqdm(loader, desc=f"Train E{epoch}", leave=False)

    total_loss = 0.0
    use_cuda = DEVICE.startswith("cuda")
    autocast_ctx = torch.cuda.amp.autocast if use_cuda else torch.cpu.amp.autocast

    batch_idx = -1
    for i, (masked_in, gt_rgb, gt_ab) in enumerate(loop):
        batch_idx = i
        masked_in = masked_in.to(DEVICE, non_blocking=True)
        gt_rgb = gt_rgb.to(DEVICE, non_blocking=True)
        gt_ab = gt_ab.to(DEVICE, non_blocking=True)

        # clear some cache aggressively (useful on 4GB)
        torch.cuda.empty_cache()

        with autocast_ctx():
            out_recon, out_color = model(masked_in)

        out_recon_f = out_recon.float()
        out_color_f = out_color.float()

        l1_rec = l1_loss(out_recon_f, gt_rgb)
        ssim_rec = 1.0 - ssim_loss(out_recon_f, gt_rgb)
        loss_recon = 0.8 * l1_rec + 0.2 * ssim_rec

        loss_color = l1_loss(out_color_f, gt_ab) * COLOR_LOSS_WEIGHT

        total_batch_loss = loss_recon + loss_color
        loss_scaled = total_batch_loss / ACCUMULATION_STEPS

        scaler.scale(loss_scaled).backward()

        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += total_batch_loss.item()

        # minimal logging per batch to avoid extra overhead
        if USE_WANDB:
            wandb.log({
                "batch/loss": total_batch_loss.item(),
                "batch/l1_rec": l1_rec.item(),
                "batch/ssim_rec": ssim_rec.item(),
                "batch/l1_color": (loss_color.item() / max(1., COLOR_LOSS_WEIGHT)),
            })

        loop.set_postfix(loss=total_batch_loss.item())

    # flush remaining grads if last iteration not divisible by accumulation steps
    if batch_idx >= 0 and (batch_idx + 1) % ACCUMULATION_STEPS != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    # occasionally log one example image (saves memory/time)
    if USE_WANDB and (epoch % log_images_every == 0):
        try:
            # take a single batch from loader (safe small)
            masked_in, gt_rgb, gt_ab = next(iter(loader))
            masked_in = masked_in.to(DEVICE)
            with torch.no_grad():
                recon, color = model(masked_in)
            # log first image of batch
            wandb.log({
                "example/input": wandb.Image(masked_in[0].cpu(), caption="masked_input"),
                "example/recon": wandb.Image(recon[0].cpu(), caption="recon"),
                "example/color": wandb.Image(color[0].cpu(), caption="color_ab"),
                "example/gt": wandb.Image(gt_rgb[0].cpu(), caption="gt_rgb"),
            })
        except Exception as e:
            print("Image logging skipped due to:", e)

    return total_loss / max(1, len(loader))


def validate_one_epoch(model, loader, l1_loss, ssim_loss, epoch):
    model.eval()
    loop = tqdm(loader, desc=f"Val E{epoch}", leave=False)

    total_val_loss = 0.0
    use_cuda = DEVICE.startswith("cuda")
    autocast_ctx = torch.cuda.amp.autocast if use_cuda else torch.cpu.amp.autocast

    with torch.no_grad():
        for masked_in, gt_rgb, gt_ab in loop:
            masked_in = masked_in.to(DEVICE)
            gt_rgb = gt_rgb.to(DEVICE)
            gt_ab = gt_ab.to(DEVICE)

            with autocast_ctx():
                out_recon, out_color = model(masked_in)

            out_recon_f = out_recon.float()
            out_color_f = out_color.float()

            l1_rec = l1_loss(out_recon_f, gt_rgb)
            ssim_rec = 1.0 - ssim_loss(out_recon_f, gt_rgb)
            loss_recon = 0.8 * l1_rec + 0.2 * ssim_rec

            loss_color = l1_loss(out_color_f, gt_ab) * COLOR_LOSS_WEIGHT

            total_val_loss += (loss_recon + loss_color).item()

    return total_val_loss / max(1, len(loader))


# -------------------------
# Main
# -------------------------
def main():
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = setup_dataloaders()
    model, optimizer = build_model_and_optimizer()

    # create losses and scaler
    l1_loss = nn.L1Loss().to(DEVICE)
    ssim_loss = SSIM(data_range=1.0).to(DEVICE)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.startswith("cuda"))

    start_epoch, best_val_loss = load_checkpoint(model, optimizer, scaler)

    print("Training start - device:", DEVICE)
    try:
        for epoch in range(start_epoch, EPOCHS):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, l1_loss, ssim_loss, scaler, epoch)
            val_loss = validate_one_epoch(model, val_loader, l1_loss, ssim_loss, epoch)

            print(f"Epoch {epoch+1}/{EPOCHS} - train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | time: {time.time()-t0:.1f}s")

            if USE_WANDB:
                wandb.log({
                    "epoch/train_loss": train_loss,
                    "epoch/val_loss": val_loss,
                    "epoch": epoch + 1
                })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("New best model - saving.")
                torch.save(model.state_dict(), BEST_MODEL_PATH)

            save_checkpoint(epoch, model, optimizer, best_val_loss, scaler)

    except KeyboardInterrupt:
        print("Interrupted by user — saving latest checkpoint...")
        save_checkpoint(epoch, model, optimizer, best_val_loss, scaler)
        raise
    except Exception as e:
        print("Exception during training:", e)
        save_checkpoint(epoch, model, optimizer, best_val_loss, scaler)
        raise

    print("Training finished!")

if __name__ == "__main__":
    main()
