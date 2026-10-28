import torch
import torch.nn as nn
import timm
from lora import LoRALayer


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        # Upsample
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        # Use GroupNorm (better than BN for small batches)
        gn_groups = 8
        if out_ch % gn_groups != 0:
            gn_groups = 4 if out_ch % 4 == 0 else 1

        self.res = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(gn_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(gn_groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_up = self.up(x)
        return self.res(x_up) + x_up


class DinoLoRADualDecoder(nn.Module):
    def __init__(self, lora_rank=32):
        super().__init__()

        # Load DINOv2 ViT-S/14
        self.encoder = timm.create_model(
            "vit_small_patch14_dinov2",
            pretrained=True,
            img_size=224
        )

        # Freeze encoder fully first
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Insert LoRA into all Linear layers
        self.inject_lora(self.encoder, rank=lora_rank)

        # Unfreeze last block for adaptation
        for p in self.encoder.blocks[-1].parameters():
            p.requires_grad = True

        # Neck (conv to spatial map)
        self.neck = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=1),
            nn.GroupNorm(8, 384),
            nn.ReLU(inplace=True)
        )

        # Shared decoder
        self.shared_decoder = nn.Sequential(
            DecoderBlock(384, 192),
            DecoderBlock(192, 96),
        )

        # Reconstruction (RGB) branch
        self.recon_branch = self._make_branch(out_ch=3, final_act=nn.Sigmoid())

        # Colorization (ab) branch
        self.color_branch = self._make_branch(out_ch=2, final_act=nn.Tanh())

    def _make_branch(self, out_ch, final_act):
        return nn.Sequential(
            DecoderBlock(96, 48),
            DecoderBlock(48, 24),
            nn.Conv2d(24, out_ch, kernel_size=1),
            final_act,
        )

    def inject_lora(self, module, rank):
        # Recursively replace Linear layers with LoRA-wrapped layers
        for name, child in module._modules.items():
            if isinstance(child, nn.Linear):
                module._modules[name] = LoRALayer(child, rank)
            elif len(list(child.children())) > 0:
                self.inject_lora(child, rank)

    def forward(self, x):
        x = self.encoder.forward_features(x)  # (B, 197, 384)
        x = x[:, 1:, :]  # remove CLS token → (B, 196, 384)

        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        x = self.neck(x)
        x = self.shared_decoder(x)

        recon = self.recon_branch(x)
        color = self.color_branch(x)

        recon = nn.functional.interpolate(recon, size=(224, 224), mode="bilinear")
        color = nn.functional.interpolate(color, size=(224, 224), mode="bilinear")

        return recon, color
