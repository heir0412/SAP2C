"""Dual-stream feature fusion for the Mask Generation Block (MGB)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskFusionMLP(nn.Module):
    """Fuse ancient-painting and realistic-image features at every pixel.

    Each stream is projected to a 256-dimensional representation.  The MLP
    then receives ``[M_anc, F_anc, M_real, F_real]`` and uses hidden layers of
    256, 128, and 64 units followed by a sigmoid mask prediction.
    """

    def __init__(self, feature_channels=256, hidden_channels=(256, 128, 64)):
        super().__init__()
        if tuple(hidden_channels) != (256, 128, 64):
            raise ValueError("The released SAP2C MGB uses hidden layers (256, 128, 64).")
        self.feature_channels = feature_channels
        self.ancient_projection = nn.LazyConv2d(feature_channels, kernel_size=1)
        self.realistic_projection = nn.LazyConv2d(feature_channels, kernel_size=1)
        self.fc1 = nn.Linear(2 + 2 * feature_channels, hidden_channels[0])
        self.fc2 = nn.Linear(hidden_channels[0], hidden_channels[1])
        self.fc3 = nn.Linear(hidden_channels[1], hidden_channels[2])
        self.output = nn.Linear(hidden_channels[2], 1)

    @staticmethod
    def _align(tensor, size):
        if tensor.ndim != 4:
            raise ValueError("MGB inputs must have shape [B, C, H, W].")
        return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)

    def forward(self, m_anc, f_anc, m_real, f_real):
        size = m_anc.shape[-2:]
        m_anc = self._align(m_anc, size)
        m_real = self._align(m_real, size)
        f_anc = self.ancient_projection(self._align(f_anc, size))
        f_real = self.realistic_projection(self._align(f_real, size))
        joint = torch.cat([m_anc, f_anc, m_real, f_real], dim=1)
        joint = joint.permute(0, 2, 3, 1)
        joint = F.relu(self.fc1(joint), inplace=True)
        joint = F.relu(self.fc2(joint), inplace=True)
        joint = F.relu(self.fc3(joint), inplace=True)
        logits = self.output(joint).permute(0, 3, 1, 2)
        return logits


def dice_loss_from_logits(logits, target, eps=1e-6):
    """Soft Dice loss matching the manuscript's squared-denominator form."""
    probability = torch.sigmoid(logits)
    target = target.float().clamp(0, 1)
    intersection = (probability * target).sum(dim=(2, 3))
    denominator = probability.square().sum(dim=(2, 3)) + target.square().sum(dim=(2, 3))
    return 1.0 - ((2.0 * intersection + eps) / (denominator + eps)).mean()


def fusion_loss(logits, target, loss_type="bce_dice", lambda_bce=1.0, lambda_dice=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
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
