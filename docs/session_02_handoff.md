# Session 2 Handoff — LiDAR Projection & Depth Map Generation
**Project:** ECE 5424 Advanced ML Capstone — SNN vs CNN for LiDAR-Event Camera Fusion
**Student:** Victor Velasquez | Virginia Tech Spring 2026
**Session completed:** 2026-04-20
**Notebook produced:** `notebooks/04_lidar_projection.ipynb`

---

## What Session 2 Accomplished

1. Parsed DSEC calibration files and extracted the LiDAR→camera transform
2. Projected all 350 LiDAR sweeps into the 640×480 camera frame
3. Saved 350 depth maps as `.npy` files
4. Built and smoke-tested `DSECFusionDataset` — the class Sessions 3–5 will use for training

---

## Calibration Structure (critical for Session 3+)

### cam_to_lidar.yaml

```
T_lidar_camRect1:          ← single 4×4 homogeneous matrix (NOT separate R and t keys)
- [R00, R01, R02, tx]
- [R10, R11, R12, ty]
- [R20, R21, R22, tz]
- [0,   0,   0,   1 ]
```

**Naming convention:** `T_lidar_camRect1` maps points FROM `camRect1` INTO the lidar frame.
To project LiDAR points INTO the camera frame, you need the **inverse**:

```python
T_lc   = np.array(cal_lidar['T_lidar_camRect1'])   # (4,4)
R_lc   = T_lc[:3, :3]
t_lc   = T_lc[:3,  3]
R      = R_lc.T            # inverse rotation
t      = -R_lc.T @ t_lc   # inverse translation
# Then: p_cam = R @ p_lidar + t
```

**Extracted values:**
```
R (LiDAR → camRect1):
[[ 0.00650225 -0.9996294  -0.02643434]
 [ 0.00164144  0.02644554 -0.99964891]
 [ 0.99997751  0.00645658  0.00181279]]

t: [ 0.24631  -0.22240  -0.44925 ] metres
```

### cam_to_cam.yaml — intrinsics structure

Camera intrinsics are nested under `intrinsics → <camera_name> → camera_matrix`.
Format: `[fx, fy, cx, cy]` (flat list of 4 values, NOT a matrix).

The relevant cameras:
| Key | Type | Resolution | Rectified |
|-----|------|------------|-----------|
| `cam0` | event (left) | 640×480 | No |
| `cam1` | frame (left) | 1440×1080 | No |
| `camRect0` | event (left) | 640×480 | **Yes** |
| `camRect1` | frame (left) | 1440×1080 | **Yes** ← LiDAR calibrated to this |

**camRect1 native intrinsics (1440×1080):**
```
fx = fy = 1150.8944
cx = 723.4334
cy = 572.1022
```

---

## Critical Bug Found and Fixed: Direct Projection vs. Resize

### What went wrong first
Initial approach: project at 1440×1080 → nearest-neighbour downscale to 640×480.

```
4,112 points land in 1440×1080 → only 867 survive the downscale (78% data loss)
```

Nearest-neighbour downscaling collapses 4–5 LiDAR points per output pixel into one,
discarding the rest. The depth map had only 0.3% fill (867 / 307,200 pixels).

### The fix: scaled intrinsics, project directly at 640×480

Scale the intrinsic matrix to the target resolution, then project in one step:

```python
scale_u = 640 / 1440   # ≈ 0.4444
scale_v = 480 / 1080   # ≈ 0.4444

fx_s = fx * scale_u    # 1150.8944 * 0.4444 = 511.509
fy_s = fy * scale_v    # 1150.8944 * 0.4444 = 511.509
cx_s = cx * scale_u    # 723.4334  * 0.4444 = 321.526
cy_s = cy * scale_v    # 572.1022  * 0.4444 = 254.268

# Overwrite for all downstream use
fx, fy, cx, cy = fx_s, fy_s, cx_s, cy_s
```

Result: **4,114 non-zero pixels** (4.7× more than the resize approach).

### Final projection function

```python
def project_lidar_to_depth(points, R, t, fx, fy, cx, cy, W, H):
    pts_xyz = points[:, :3]
    pts_cam = (R @ pts_xyz.T).T + t          # LiDAR → camera coords
    in_front = pts_cam[:, 2] > 0.1           # discard points behind camera
    pts_cam  = pts_cam[in_front]
    Z = pts_cam[:, 2]
    u = (fx * pts_cam[:, 0] / Z + cx).astype(np.int32)
    v = (fy * pts_cam[:, 1] / Z + cy).astype(np.int32)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, Z = u[inside], v[inside], Z[inside]
    depth_map = np.zeros((H, W), dtype=np.float32)
    order = np.argsort(Z)[::-1]              # far→near so near overwrites
    depth_map[v[order], u[order]] = Z[order]
    return depth_map
```

Call with scaled intrinsics and target resolution directly:
```python
depth_map = project_lidar_to_depth(points, R, t, fx, fy, cx, cy, W=640, H=480)
```

---

## LiDAR Characteristics (VLP-16)

- **Sensor:** Velodyne VLP-16 (confirmed from ~25k points/sweep)
- **Beams:** 16 vertical — produces 16 horizontal scan lines in the projected image
- **Points/sweep:** ~24,952
- **In front of camera:** ~49.2% (expected — 360° LiDAR, camera covers forward half)
- **In camera FOV:** ~33.5% of forward-facing points (~4,112 per sweep)
- **Depth range:** 4.8 – 103.8 m

The projected depth map looks sparse (16 dotted horizontal lines). This is normal for
VLP-16. Depth maps are NOT dense — zeros mean no LiDAR return, not zero distance.

---

## Depth Maps

- **Location:** `dsec-data/zurich_city_04_a/depth_maps/` (local only, gitignored)
- **Format:** `{frame_idx:04d}.npy`, shape `(480, 640)`, float32
- **Units:** metres. Pixel value 0 = no LiDAR return
- **Count:** 350 files (0000.npy – 0349.npy, matching `frame_idx` in pairs JSON)
- **Typical non-zero pixels:** ~3,000–4,100 per frame

---

## Semantic Labels — Actual Path Structure

The semantic labels are NOT at the top of `zurich_city_04_a_semantic/`.
The actual structure is:

```
zurich_city_04_a_semantic/
└── zurich_city_04_a/
    ├── 11classes/          ← USE THIS (701 PNGs: 000000.png – 000700.png)
    ├── 19classes/          ← too many classes for 3-layer networks
    └── zurich_city_04_a_semantic_timestamps.txt
```

**Use 11classes.** 19 classes is too granular for shallow 3-layer networks.

**File count:** 701 PNGs total. Our `frame_idx` values (0–349) index the first 350 files.
Files 350–700 correspond to frames outside our LiDAR overlap window — never loaded.

**In the notebook, LABEL_DIR is:**
```python
LABEL_DIR = SEMANTIC_DIR / "zurich_city_04_a" / "11classes"
# where SEMANTIC_DIR = SEQ_DIR / f"{SEQUENCE}_semantic"
```

### Bug fixed: label resolution mismatch

DSEC semantic PNGs are **640×440**, not 640×480. The depth maps are 480 pixels tall.
Fixed in `dsec_dataset.py` with nearest-neighbour resize (bilinear would corrupt class IDs):

```python
if label_np.shape != (target_h, target_w):
    label_np = cv2.resize(
        label_np, (target_w, target_h), interpolation=cv2.INTER_NEAREST
    )
```

---

## Semantic Classes

| Stat | Value |
|------|-------|
| Total class IDs | **11** (IDs 0–10) |
| Classes in a single frame | 7–9 (not all 11 appear in every frame — normal) |
| All 11 IDs seen across batch | ✅ confirmed |

These 11 IDs are what the segmentation head must predict. The CNN and SNN output
feature maps feed into a head with **11 output channels**, one per class.

---

## DSECFusionDataset — Final State

**File:** `src/datasets/dsec_dataset.py`

```python
dataset = DSECFusionDataset(
    pairs_json    = REPO_ROOT / "data" / "zurich_city_04_a_pairs.json",
    depth_map_dir = SEQ_DIR / "depth_maps",
    semantic_dir  = SEQ_DIR / "zurich_city_04_a_semantic" / "zurich_city_04_a" / "11classes",
)
```

**Output per sample:**

| Tensor | Shape | dtype | Notes |
|--------|-------|-------|-------|
| `time_surface` | `(2, 480, 640)` | float32 | **zeros placeholder** — Session 3 replaces this |
| `depth` | `(1, 480, 640)` | float32 | LiDAR projection, metres |
| `label` | `(480, 640)` | int64 | Class IDs 0–10 |

**DataLoader confirmed working** with batch_size=4, shuffle=True, num_workers=0.

---

## Files Produced This Session

| File | Location | Notes |
|------|----------|-------|
| `04_lidar_projection.ipynb` | `notebooks/` | Session 2 notebook |
| `dsec_dataset.py` | `src/datasets/` | Dataset class with label resize fix |
| `__init__.py` | `src/datasets/` | Package init |
| `depth_maps/*.npy` | `dsec-data/.../depth_maps/` | **Local only, gitignored** |
| `lidar_projection_verification.png` | `docs/figures/` | First alignment check |
| `lidar_projection_direct_v2.png` | `docs/figures/` | 4.7× denser direct projection |
| `depth_map_grid.png` | `docs/figures/` | 12-frame depth map grid |
| `depth_overlay_strip.png` | `docs/figures/` | RGB+depth strip across sequence |
| `.gitignore` updated | repo root | Added `**/depth_maps/` |

---

## All Known Quirks (Sessions 1 + 2 combined)

| Issue | Root cause | Fix |
|-------|-----------|-----|
| `OSError` reading `events.h5` | BLOSC compression | `import hdf5plugin` before `import h5py` |
| Event timestamps don't overlap LiDAR | DSEC stores relative timestamps | Add `t_offset` from root of `events.h5` |
| `ImportError: deserialize_cdr` | rosbags ≥ 0.9 removed this | `typestore.deserialize_ros1(rawdata, msgtype)` |
| `T_lidar_camRect1` isn't separate R and t | It's a 4×4 matrix | `np.array(cal['T_lidar_camRect1'])` → invert |
| Only 867 non-zero pixels in depth map | Resize discards 78% of LiDAR data | Use scaled intrinsics, project directly at 640×480 |
| Label shape (440, 640) ≠ depth (480, 640) | DSEC semantic PNGs are 640×440 | `cv2.resize(..., INTER_NEAREST)` in dataset class |
| `FileNotFoundError` for semantic PNGs | Labels are 2 subdirs deep | Path: `*_semantic/zurich_city_04_a/11classes/` |

---

## Session 3 — What to Build Next

**Notebook:** `notebooks/05_snn_branch.ipynb`
**Goal:** Replace the `time_surface` zeros placeholder with real event-based time surfaces,
build the 3-layer LIF SNN, and produce a `(C, 480, 640)` feature map.

### Step 1 — Event window slicing

For each pair in `zurich_city_04_a_pairs.json`, slice events around `frame_ts`:

```python
# Window: [frame_ts - 50ms, frame_ts]  (one LiDAR period before the label)
window_us = 50_000   # 50 ms in microseconds
t_start = pair['frame_ts'] - window_us
t_end   = pair['frame_ts']

# Load events from events.h5 using ms_to_idx for fast slicing
# Remember: import hdf5plugin before h5py
# Remember: event timestamps = events/t + t_offset (absolute µs)
```

### Step 2 — Time surface construction

Two-channel time surface (positive + negative polarity):
- Channel 0: timestamp of most recent positive event at each pixel
- Channel 1: timestamp of most recent negative event at each pixel
- Normalize to [0, 1] within the window

Shape: `(2, 480, 640)` float32 — this replaces the zeros in `DSECFusionDataset`.

Note: events are in the **event camera (camRect0)** coordinate frame — no reprojection needed.
Use the rectification map (`rectify_map.h5`) if working with raw (non-rectified) events.

### Step 3 — LIF SNN architecture (3 layers)

```
Input: (2, 480, 640) time surface
→ Conv2d(2, 16, 3, padding=1) + LIF neuron
→ Conv2d(16, 32, 3, padding=1, stride=2) + LIF neuron   [240×320]
→ Conv2d(32, 64, 3, padding=1, stride=2) + LIF neuron   [120×160]
→ Feature map: (64, 120, 160)
```

Use `snntorch` (already in `requirements.txt`) for LIF neurons.
Output feature map feeds the Smart Gate fusion layer in Session 5.

### Step 4 — Forward pass test

Run one batch through the SNN, confirm output shape is `(B, 64, 120, 160)`.
Save the SNN module to `src/models/snn_branch.py`.

### What NOT to do in Session 3
- Do not build the CNN branch (Session 4)
- Do not build the fusion layer (Session 5)
- Do not start training
- Do not process `zurich_city_00_a`

---

## Session 4 — CNN Branch (after Session 3)

**Notebook:** `notebooks/06_cnn_branch.ipynb`

```
Input: (1, 480, 640) depth map
→ Conv2d(1, 16, 3, padding=1) + BN + ReLU
→ Conv2d(16, 32, 3, padding=1, stride=2) + BN + ReLU   [240×320]
→ Conv2d(32, 64, 3, padding=1, stride=2) + BN + ReLU   [120×160]
→ Feature map: (64, 120, 160)
```

Both branches must output the **same shape** `(B, 64, 120, 160)` for the Smart Gate fusion.

---

## Session 5 — Fusion + Training

**Smart Gate fusion:**
```
snn_features: (B, 64, 120, 160)  → sigmoid → attention map (values 0–1)
cnn_features: (B, 64, 120, 160)
fused = attention_map * cnn_features   (elementwise multiply)
→ ConvTranspose upsample to (B, 11, 480, 640)
→ CrossEntropyLoss against label (B, 480, 640)
```

**Training setup:**
- Optimizer: Adam, lr=1e-3
- Loss: CrossEntropyLoss (handles class imbalance better than MSE)
- Batch size: 4 (depth maps + time surfaces fit in GPU memory)
- Split: 280 train / 70 val (80/20 from 350 pairs)

---

## Completed Notebook Inventory

| Notebook | Status | Content |
|----------|--------|---------|
| `01_time_surface_dsec.ipynb` | ✅ complete | DVS event → time surface generation |
| `02_disparity_eda.ipynb` | ✅ complete | Disparity EDA |
| `03_lidar_extraction_alignment.ipynb` | ✅ complete | LiDAR extraction + temporal matching |
| `04_lidar_projection.ipynb` | ✅ complete | LiDAR → camera projection + depth maps |
| `05_snn_branch.ipynb` | 🔲 Session 3 | Time surface + LIF SNN branch |
| `06_cnn_branch.ipynb` | 🔲 Session 4 | CNN depth encoder branch |
| `07_fusion_training.ipynb` | 🔲 Session 5 | Smart Gate fusion + training + eval |
