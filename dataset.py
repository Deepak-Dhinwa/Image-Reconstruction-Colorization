import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.color import rgb2lab


class ReconColorDataset(Dataset):
    ALLOWED_EXT = ('.jpg', '.jpeg', '.png')

    def __init__(self, image_dir, image_size=224, mask_ratio=0.50, patch_size=14, remove_invalid=False):
        self.image_dir = image_dir
        self.image_size = image_size
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

        # List image files
        self.image_files = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(self.ALLOWED_EXT)
        ]

        # Remove corrupt images safely
        if remove_invalid:
            good = []
            for p in self.image_files:
                try:
                    Image.open(p).convert("RGB")
                    good.append(p)
                except Exception:
                    print(f"⚠️ Warning: Removing invalid image: {p}")
            self.image_files = good

        assert len(self.image_files) > 0, f"No valid images found in {image_dir}"

        # Input transform — normalize like DINOv2 expects
        self.transform_input = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Ground truth RGB transform — no normalization
        self.transform_gt = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_pil = Image.open(img_path).convert("RGB")

        # Ground truth RGB
        gt_rgb = self.transform_gt(img_pil)  # [0,1]

        # LAB full image
        img_lab = rgb2lab(gt_rgb.permute(1, 2, 0).numpy()).astype("float32")

        # AB channels → [-1,1]
        gt_ab = torch.from_numpy(img_lab[:, :, 1:]).permute(2, 0, 1) / 128.0

        # 3-channel grayscale input
        img_gray = img_pil.convert("L").convert("RGB")
        input_norm = self.transform_input(img_gray)

        # Create patch mask
        p = self.patch_size
        S = self.image_size // p
        total_patches = S * S
        masked = int(total_patches * self.mask_ratio)

        perm = np.random.permutation(total_patches)
        mask_1d = torch.ones(total_patches, dtype=torch.float32)
        mask_1d[perm[:masked]] = 0.0

        mask_2d = mask_1d.view(S, S)
        mask_up = mask_2d.repeat_interleave(p, 0).repeat_interleave(p, 1)
        mask_up = mask_up.unsqueeze(0)  # (1,H,W)

        degraded_input = input_norm * mask_up

        return degraded_input, gt_rgb, gt_ab
