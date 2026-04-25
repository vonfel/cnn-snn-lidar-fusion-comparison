import json
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


def build_time_surface(t_abs, x_all, y_all, p_all,
                       frame_ts, window_us=50_000, H=480, W=640):
    """
    Build a 2-channel time surface for a given frame timestamp.

    For each pixel, stores the normalized timestamp of the most recent
    event within the window [frame_ts - window_us, frame_ts].
      Channel 0 = ON  polarity (p=1)
      Channel 1 = OFF polarity (p=0)

    Normalization: active pixel value = (t_event - t_start) / window_us
    so values are in [0, 1]. Pixels with no event in the window stay at 0.

    Args:
        t_abs      : (N,) int64  absolute event timestamps in µs
        x_all      : (N,) uint16 pixel x coordinates
        y_all      : (N,) uint16 pixel y coordinates
        p_all      : (N,) bool/uint8 polarity (1=ON, 0=OFF)
        frame_ts   : int   frame timestamp in absolute µs
        window_us  : int   time window width in µs (default 50 ms)
        H, W       : int   output spatial dimensions

    Returns:
        time_surface : (2, H, W) float32
    """
    t_start = frame_ts - window_us
    t_end = frame_ts

    # Binary search — avoids iterating over all 358M events per sample
    idx_start = np.searchsorted(t_abs, t_start, side='left')
    idx_end = np.searchsorted(t_abs, t_end, side='right')

    if idx_end <= idx_start:
        return np.zeros((2, H, W), dtype=np.float32)

    t_win = t_abs[idx_start:idx_end]
    x_win = x_all[idx_start:idx_end].astype(np.int32)
    y_win = y_all[idx_start:idx_end].astype(np.int32)
    p_win = p_all[idx_start:idx_end]

    # Safety clip to image bounds
    valid = (x_win >= 0) & (x_win < W) & (y_win >= 0) & (y_win < H)
    t_win = t_win[valid]
    x_win = x_win[valid]
    y_win = y_win[valid]
    p_win = p_win[valid]

    ts = np.zeros((2, H, W), dtype=np.float32)

    # ON events (p=1)
    on_mask = p_win == 1
    ts[0, y_win[on_mask], x_win[on_mask]] = t_win[on_mask].astype(np.float32)

    # OFF events (p=0)
    off_mask = p_win == 0
    ts[1, y_win[off_mask], x_win[off_mask]] = t_win[off_mask].astype(np.float32)

    # Normalize to [0, 1] within the window; zero pixels stay zero
    for c in range(2):
        active = ts[c] > 0
        if active.any():
            ts[c][active] = (ts[c][active] - t_start) / window_us

    return ts


class DSECFusionDataset(Dataset):
    """
    Loads matched triplets for one DSEC sequence.

    Each sample returns:
      time_surface : (2, 480, 640) float32  -- ON/OFF event polarity channels
                     Normalized timestamps of the most recent event per pixel
                     within a 50 ms window ending at frame_ts.
      depth_map    : (1, 480, 640) float32  -- projected LiDAR depth (metres)
                     0 means no LiDAR return at that pixel.
      label        : (480, 640)    int64    -- semantic class IDs 0-10

    The event arrays (t_abs, x_all, y_all, p_all) must be pre-loaded by the
    caller and passed in. Loading them once at construction avoids re-opening
    the BLOSC-compressed events.h5 (358M events) on every __getitem__ call.

    Args:
        pairs_json    : path to zurich_city_04_a_pairs.json
        depth_map_dir : folder containing {frame_idx:04d}.npy depth maps
        semantic_dir  : folder containing semantic label PNGs (11classes/)
        t_abs         : (N,) int64  absolute event timestamps in µs
        x_all         : (N,) pixel x coordinates
        y_all         : (N,) pixel y coordinates
        p_all         : (N,) polarity (1=ON, 0=OFF)
    """

    def __init__(self, pairs_json, depth_map_dir, semantic_dir,
                 t_abs, x_all, y_all, p_all):
        with open(pairs_json) as f:
            self.pairs = json.load(f)

        self.depth_map_dir = Path(depth_map_dir)
        self.semantic_dir = Path(semantic_dir)

        # Sort label files so index 0 = earliest frame
        self.label_files = sorted(self.semantic_dir.glob("*.png"))

        if len(self.label_files) == 0:
            raise FileNotFoundError(
                f"No PNG files found in semantic_dir: {self.semantic_dir}"
            )

        # Event arrays pre-loaded by caller
        self.t_abs = t_abs
        self.x_all = x_all
        self.y_all = y_all
        self.p_all = p_all

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        frame_idx = pair['frame_idx']

        # --- Time surface (SNN branch input) ---
        ts_np = build_time_surface(
            self.t_abs, self.x_all, self.y_all, self.p_all,
            pair['frame_ts']
        )
        time_surface = torch.from_numpy(ts_np)  # (2, 480, 640) float32

        # --- Depth map (CNN branch input) ---
        # Shape on disk: (H, W) float32, units = metres, 0 = no return
        depth_np = np.load(self.depth_map_dir / f"{frame_idx:04d}.npy")
        depth = torch.from_numpy(depth_np).unsqueeze(0)  # (1, 480, 640)

        # --- Semantic label (ground truth) ---
        # Each pixel value is an integer class ID; read as grayscale
        label_np = cv2.imread(
            str(self.label_files[frame_idx]), cv2.IMREAD_GRAYSCALE
        )
        if label_np is None:
            raise FileNotFoundError(
                f"Could not read label file: {self.label_files[frame_idx]}"
            )

        # DSEC semantic PNGs are 640×440; depth maps are 640×480.
        # Must use nearest-neighbour: bilinear would blend class IDs and create
        # invalid fractional values (e.g. class 2.7) that corrupt the loss.
        target_h, target_w = depth_np.shape[0], depth_np.shape[1]
        if label_np.shape != (target_h, target_w):
            label_np = cv2.resize(
                label_np, (target_w, target_h),
                interpolation=cv2.INTER_NEAREST
            )

        label = torch.from_numpy(label_np).long()  # (480, 640) int64

        return time_surface, depth, label

    def get_num_classes(self):
        """Count unique class IDs in the first label image."""
        label_np = cv2.imread(
            str(self.label_files[0]), cv2.IMREAD_GRAYSCALE
        )
        return int(len(np.unique(label_np)))


# ---------------------------------------------------------------------------
# Standalone smoke test
# Run with: python src/datasets/dsec_dataset.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT))

    import hdf5plugin  # must come before h5py
    import h5py
    from config import DATA_ROOT

    SEQ = "zurich_city_04_a"
    SEQ_DIR = Path(DATA_ROOT) / SEQ
    EVENTS_PATH = SEQ_DIR / f"{SEQ}_events_left" / "events.h5"

    print("Loading events ...")
    with h5py.File(EVENTS_PATH, 'r') as f:
        t_offset = int(f['t_offset'][()])
        t_abs = f['events']['t'][:].astype(np.int64) + t_offset
        x_all = f['events']['x'][:]
        y_all = f['events']['y'][:]
        p_all = f['events']['p'][:]
    print(f"  {len(t_abs):,} events loaded")

    dataset = DSECFusionDataset(
        pairs_json=REPO_ROOT / "data" / "zurich_city_04_a_pairs.json",
        depth_map_dir=SEQ_DIR / "depth_maps",
        semantic_dir=SEQ_DIR / f"{SEQ}_semantic" / SEQ / "11classes",
        t_abs=t_abs, x_all=x_all, y_all=y_all, p_all=p_all,
    )

    print(f"Dataset size: {len(dataset)}")
    ts, depth, label = dataset[0]
    print(f"Time surface : {ts.shape}  non-zero={int((ts > 0).sum())}")
    print(f"Depth map    : {depth.shape}  non-zero={int((depth > 0).sum())}")
    print(f"Label        : {label.shape}  classes={label.unique().tolist()}")
