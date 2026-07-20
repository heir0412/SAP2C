import torch
import torch.nn as nn


class MaskFusionMLP(nn.Module):
    """Per-pixel MLP for fusing LISA, DINOv3, and semantic mask cues."""

    def __init__(self, in_channels=4, hidden_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


def dice_loss_from_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    target = target.float().clamp(0, 1)
    inter = (prob * target).sum(dim=(2, 3))
    denom = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return 1.0 - ((2.0 * inter + eps) / denom).mean()


def fusion_loss(logits, target, loss_type="bce_dice", lambda_bce=2.0, lambda_dice=0.5):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss_from_logits(logits, target)
    if loss_type == "bce":
        total = bce
    elif loss_type == "dice":
        total = dice
    elif loss_type == "bce_dice":
        total = lambda_bce * bce + lambda_dice * dice
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    return total, bce, dice
