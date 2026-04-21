import json
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class DSECFusionDataset(Dataset):
    """
    Loads matched triplets for one DSEC sequence.

    Each sample is a tuple of three tensors:
      - time_surface : (2, H, W) float32  -- event camera input for the SNN branch
                       Positive and negative polarity channels.
                       Currently a zeros placeholder; Session 3 fills this in.
      - depth_map    : (1, H, W) float32  -- LiDAR depth projection for the CNN branch
                       Pixel value = distance in metres; 0 means no LiDAR return.
      - label        : (H, W)    int64    -- semantic segmentation ground truth
                       Each pixel value is a class index (0 = background, etc.)

    Args:
        pairs_json    : path to zurich_city_04_a_pairs.json
        depth_map_dir : folder containing {frame_idx:04d}.npy depth maps
        semantic_dir  : folder containing semantic label PNGs
    """

    def __init__(self, pairs_json, depth_map_dir, semantic_dir):
        with open(pairs_json) as f:
            self.pairs = json.load(f)

        self.depth_map_dir = Path(depth_map_dir)
        self.semantic_dir  = Path(semantic_dir)

        # Sort label files so index 0 = earliest frame
        self.label_files = sorted(self.semantic_dir.glob("*.png"))

        if len(self.label_files) == 0:
            raise FileNotFoundError(
                f"No PNG files found in semantic_dir: {self.semantic_dir}"
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair      = self.pairs[idx]
        frame_idx = pair["frame_idx"]

        #  Depth map (CNN branch input)
        # Shape on disk: (H, W) float32, units = metres, 0 = no return
        depth_np = np.load(self.depth_map_dir / f"{frame_idx:04d}.npy")
        # Add channel dim so shape becomes (1, H, W) — standard conv input format
        depth = torch.from_numpy(depth_np).unsqueeze(0)

        # Semantic label (ground truth)
        # Each pixel value is an integer class ID, so read as grayscale
        label_np = cv2.imread(
            str(self.label_files[frame_idx]), cv2.IMREAD_GRAYSCALE
        )
        if label_np is None:
            raise FileNotFoundError(
                f"Could not read label file: {self.label_files[frame_idx]}"
            )

        # DSEC semantic PNGs are 640×440 — resize to match depth map (480×640).
        # Must use nearest-neighbour: bilinear would blend class IDs and create
        # invalid fractional values (e.g. class 2.7) that corrupt the loss.
        target_h, target_w = depth_np.shape[0], depth_np.shape[1]
        if label_np.shape != (target_h, target_w):
            label_np = cv2.resize(
                label_np, (target_w, target_h), interpolation=cv2.INTER_NEAREST
            )

        label = torch.from_numpy(label_np).long()  # (H, W), dtype int64

        # Time surface (SNN branch input) — placeholder
        # 2 channels: positive polarity and negative polarity event frames
        # Will be replaced in Session 3 with real time surface tensors
        H, W = depth.shape[1], depth.shape[2]
        time_surface = torch.zeros(2, H, W, dtype=torch.float32)

        return time_surface, depth, label

    def get_num_classes(self):
        """Count unique class IDs across the first label image."""
        label_np = cv2.imread(
            str(self.label_files[0]), cv2.IMREAD_GRAYSCALE
        )
        return int(len(np.unique(label_np)))


# Quick standalone test
# Run with:  python src/datasets/dsec_dataset.py
if __name__ == "__main__":
    import sys

    # Add repo root to path so we can import config
    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT))
    from config import DATA_ROOT

    SEQ       = "zurich_city_04_a"
    SEQ_DIR   = Path(DATA_ROOT) / SEQ

    dataset = DSECFusionDataset(
        pairs_json    = REPO_ROOT / "data" / "zurich_city_04_a_pairs.json",
        depth_map_dir = SEQ_DIR / "depth_maps",
        semantic_dir  = SEQ_DIR / f"{SEQ}_semantic",
    )

    print(f"Dataset size    : {len(dataset)}")
    print(f"Number of classes: {dataset.get_num_classes()}")

    ts, depth, label = dataset[0]
    print(f"Time surface shape : {ts.shape}")
    print(f"Depth map shape    : {depth.shape}")
    print(f"Label shape        : {label.shape}")
    print(f"Label unique IDs   : {label.unique().tolist()}")
