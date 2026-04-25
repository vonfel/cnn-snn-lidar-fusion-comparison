import torch
import torch.nn as nn
from src.models.snn_encoder import SNNEncoder
from src.models.cnn_encoder import CNNEncoder


# ---------------------------------------------------------------------------
# Shared segmentation head (reused across all three models)
# ---------------------------------------------------------------------------

def _make_head(num_classes):
    """
    Upsample 4x from (B, 64, 120, 160) back to (B, num_classes, 480, 640).
    Two ConvTranspose2d with stride=2 each double the spatial dimensions.
    """
    return nn.Sequential(
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, num_classes, kernel_size=4, stride=2, padding=1)
    )


# ---------------------------------------------------------------------------
# Model A — Depth-only CNN baseline
# ---------------------------------------------------------------------------

class DepthOnlyCNN(nn.Module):
    """
    Model A: 3-layer CNN on projected LiDAR depth map only.
    No event data. Establishes the single-modality lower bound.

    Input:  depth  (B, 1, 480, 640)
    Output: logits (B, num_classes, 480, 640)
    """

    def __init__(self, num_classes=11):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=1)
        self.head = _make_head(num_classes)

    def forward(self, depth):
        features = self.encoder(depth)   # (B, 64, 120, 160)
        return self.head(features)       # (B, num_classes, 480, 640)


# ---------------------------------------------------------------------------
# Model B — Early Fusion CNN baseline
# ---------------------------------------------------------------------------

class EarlyFusionCNN(nn.Module):
    """
    Model B: 3-layer CNN on depth + time surface concatenated (3 channels).
    Naive fusion baseline. Shows what simple channel-level fusion can do.

    Input:  time_surface (B, 2, 480, 640)  ON/OFF event channels
            depth        (B, 1, 480, 640)  LiDAR depth
    Output: logits       (B, num_classes, 480, 640)
    """

    def __init__(self, num_classes=11):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=3)  # 1 depth + 2 time surface
        self.head = _make_head(num_classes)

    def forward(self, time_surface, depth):
        x = torch.cat([depth, time_surface], dim=1)  # (B, 3, 480, 640)
        features = self.encoder(x)                    # (B, 64, 120, 160)
        return self.head(features)                    # (B, num_classes, 480, 640)


# ---------------------------------------------------------------------------
# Model C — SNN + Smart Gate
# ---------------------------------------------------------------------------

class SmartGateModel(nn.Module):
    """
    Model C: SNN branch generates event-driven attention; CNN branch
    encodes LiDAR depth; Smart Gate gates CNN features with SNN attention.

    SNN branch: time_surface (2ch) -> LIF SNN -> attention map (64, 120, 160)
    CNN branch: depth (1ch)        -> CNN     -> feature map   (64, 120, 160)
    Fusion:     sigmoid(attention) * features  (elementwise multiply)
    Head:       4x upsample -> 11-class logits (480, 640)

    sigmoid is applied to the mean firing rate (already in [0,1]) to
    produce a smooth, differentiable gate (prevents hard feature zeroing
    at pixels with zero SNN activity).

    Where events fire -> attention ~1 -> CNN features pass at full strength.
    Where scene is static -> attention ~0 -> CNN features are suppressed.
    """

    def __init__(self, num_classes=11, beta=0.9):
        super().__init__()
        self.snn_branch = SNNEncoder(beta=beta)
        self.cnn_branch = CNNEncoder(in_channels=1)
        self.head = _make_head(num_classes)

    def forward(self, time_surface, depth, num_steps=4):
        # SNN branch: sparse, event-driven attention
        attention = self.snn_branch(time_surface, num_steps=num_steps)
        attention = torch.sigmoid(attention)        # (B, 64, 120, 160) in (0,1)

        # CNN branch: dense depth feature extraction
        features = self.cnn_branch(depth)           # (B, 64, 120, 160)

        # Smart Gate: event-driven modulation of LiDAR features
        fused = attention * features                # (B, 64, 120, 160)

        # Upsample and classify
        logits = self.head(fused)                   # (B, num_classes, 480, 640)
        return logits
