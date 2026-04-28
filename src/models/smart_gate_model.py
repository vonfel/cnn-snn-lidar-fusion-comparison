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
# Model A: Depth-only CNN baseline
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
# Model B: Early Fusion CNN baseline
# ---------------------------------------------------------------------------

class EarlyFusionCNN(nn.Module):
    """
    Model B: 3-layer CNN on depth + time surface concatenated (3 channels).
    Naive fusion baseline. Shows simple channel-level fusion capability.

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
# Model C: SNN + Smart Gate
# ---------------------------------------------------------------------------

class SmartGateModel(nn.Module):
    """
    Model C: SNN branch generates event-driven attention; CNN branch
    encodes LiDAR depth; Smart Gate gates CNN features with SNN attention.

    SNN branch: time_surface (2ch) -> LIF SNN -> attention map (64, 120, 160)
    CNN branch: depth (1ch)        -> CNN     -> feature map   (64, 120, 160)
    Fusion:     residual gate + 1x1 projection (see below)
    Head:       4x upsample -> 11-class logits (480, 640)
    """

    def __init__(self, num_classes=11, beta=0.9):
        super().__init__()
        self.snn_branch = SNNEncoder(beta=beta)
        self.cnn_branch = CNNEncoder(in_channels=1)

        # Normalize the raw SNN firing-rate map before gating.
        # Stabilises attention scale across sequences with different event
        # densities; mirrors the BN already present in CNNEncoder.
        self.gate_bn = nn.BatchNorm2d(64)

        # Learned channel recombination after the residual gate.
        # 1×1 conv keeps spatial resolution identical; output is still
        # (B, 64, 120, 160) so the shared head receives the same shape.
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.head = _make_head(num_classes)

    def forward(self, time_surface, depth, num_steps=4):
        # SNN branch: sparse, event-driven attention
        # gate_bn normalises the mean firing rate map (Fix 2) so that
        # attention scale is consistent regardless of event density.
        attention = self.snn_branch(time_surface, num_steps=num_steps)
        attention = self.gate_bn(attention)             # (B, 64, 120, 160), ~N(0,1)
        attention = torch.sigmoid(attention)            # (B, 64, 120, 160) in (0, 1)

        # CNN branch: dense depth feature extraction
        features = self.cnn_branch(depth)               # (B, 64, 120, 160)

        # Smart Gate: residual event-driven modulation
        # features * (1 + attention):
        #   attention = 0  ->  fused = features        (depth preserved)
        #   attention = 1  ->  fused = 2 * features    (depth amplified)
        fused = features + attention * features         # (B, 64, 120, 160)

        # Post-fusion channel recombination
        fused = self.fusion_proj(fused)                 # (B, 64, 120, 160)

        # Upsample and classify
        logits = self.head(fused)                       # (B, num_classes, 480, 640)
        return logits