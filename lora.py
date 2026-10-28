import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """
    Wraps a Linear layer and adds a trainable low-rank update.
    out = base(x) + (x @ A) @ B * (alpha / rank)
    """

    def __init__(self, original_layer: nn.Linear, rank: int, alpha: float = 32.0):
        super().__init__()
        self.original_layer = original_layer

        # Freeze original parameters
        if hasattr(self.original_layer, "weight"):
            self.original_layer.weight.requires_grad = False
        if hasattr(self.original_layer, "bias") and self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.rank = rank
        self.alpha = alpha

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Init
        nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base = self.original_layer(x)
        update = (x @ self.lora_A) @ self.lora_B
        update = update * (self.alpha / max(1.0, float(self.rank)))
        return base + update
