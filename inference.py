import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

from model import DinoLoRADualDecoder

# --- Configuration ---
VAL_DATA_DIR = "./coco_val2017/val2017"
MODEL_PATH = "model_BEST_smallgpu.pth"
OUTPUT_DIR = "./results"

LORA_RANK = 32
IMAGE_SIZE = 224
MASK_RATIO = 0.50
NUM_SAMPLES = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def denormalize(tensor):
    t = tensor.clone().detach().cpu()
    for c in range(3):
        t[c] = t[c] * STD[c] + MEAN[c]
    return t.clamp(0, 1)


def prepare_single_image(image_path, image_size, mask_ratio):
    img_pil = Image.open(image_path).convert("RGB")
    resize = transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC)
    img_pil = resize(img_pil)

    # GT RGB
    img_tensor_rgb = transforms.ToTensor()(img_pil)

    # Lab conversion
    img_np = np.array(img_pil) / 255.0
    img_lab = rgb2lab(img_np)

    l_channel_raw = img_lab[:, :, 0] / 100.0
    l_tensor = torch.from_numpy(l_channel_raw).float()

    # Create 3-channel grayscale L,L,L
    input_3ch = l_tensor.unsqueeze(0).repeat(3, 1, 1)

    normalize = transforms.Normalize(mean=MEAN, std=STD)
    input_normalized = normalize(input_3ch)

    # Mask
    patch_size = 14
    patches_per_side = image_size // patch_size
    num_patches = patches_per_side * patches_per_side
    num_masked = int(num_patches * mask_ratio)

    mask_indices = np.random.permutation(num_patches)[:num_masked]
    mask = torch.ones(num_patches, dtype=torch.bool)
    mask[mask_indices] = False

    mask = mask.reshape(patches_per_side, patches_per_side)
    mask_upsampled = mask.repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1).unsqueeze(0)

    masked_input = input_normalized * mask_upsampled.float()

    return masked_input.unsqueeze(0), img_tensor_rgb.unsqueeze(0), l_channel_raw


def lab_to_rgb_prediction(l_channel_raw, ab_tensor):
    ab_np = ab_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    ab_np = ab_np * 128.0

    l_np = (l_channel_raw * 100.0)[:, :, np.newaxis]

    lab_img = np.concatenate((l_np, ab_np), axis=2)
    rgb = lab2rgb(lab_img)
    return np.clip(rgb, 0, 1)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading model from {MODEL_PATH}...")
    model = DinoLoRADualDecoder(lora_rank=LORA_RANK)

    # Load weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()

    # Collect files
    all_files = [os.path.join(VAL_DATA_DIR, f)
                 for f in os.listdir(VAL_DATA_DIR)
                 if f.lower().endswith((".jpg", ".png"))]

    if len(all_files) == 0:
        print("No images found!")
        return

    sample_files = np.random.choice(all_files, NUM_SAMPLES, replace=False)

    # Autocast
    use_cuda = DEVICE.startswith("cuda")
    autocast_ctx = torch.cuda.amp.autocast if use_cuda else torch.cpu.amp.autocast

    with torch.no_grad():
        for i, img_path in enumerate(sample_files):
            masked_input, gt_rgb, l_channel_raw = prepare_single_image(
                img_path, IMAGE_SIZE, MASK_RATIO
            )
            masked_input = masked_input.to(DEVICE)

            with autocast_ctx():
                out_recon, out_color = model(masked_input)

            # Visualization
            input_disp = denormalize(masked_input.squeeze(0)).permute(1, 2, 0).numpy()
            recon_disp = np.clip(out_recon.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1)
            color_disp = lab_to_rgb_prediction(l_channel_raw, out_color)
            gt_disp = gt_rgb.squeeze(0).permute(1, 2, 0).numpy()

            fig, axes = plt.subplots(1, 4, figsize=(20, 6))

            axes[0].imshow(input_disp, cmap="gray")
            axes[0].set_title("Input (Masked)")
            axes[0].axis("off")

            axes[1].imshow(recon_disp)
            axes[1].set_title("Reconstruction")
            axes[1].axis("off")

            axes[2].imshow(color_disp)
            axes[2].set_title("Colorization")
            axes[2].axis("off")

            axes[3].imshow(gt_disp)
            axes[3].set_title("Ground Truth")
            axes[3].axis("off")

            save_path = os.path.join(OUTPUT_DIR, f"result_{i}.png")
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close()

            print(f"Saved {save_path}")

    print("Inference complete!")


if __name__ == "__main__":
    main()
