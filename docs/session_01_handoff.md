# Session 1 Handoff — LiDAR Extraction & Temporal Alignment
**Project:** ECE 5424 Advanced ML Capstone — SNN vs CNN for LiDAR-Event Camera Fusion
**Student:** Victor Velasquez | Virginia Tech Spring 2026
**Session completed:** 2026-04-19
**Notebook produced:** `notebooks/03_lidar_extraction_alignment.ipynb`

---

## Project Architecture (unchanged)

| Branch | Input | Network | Output |
|--------|-------|---------|--------|
| A (SNN) | Event camera Time Surface | 3-layer LIF SNN | Feature map |
| B (CNN) | Projected LiDAR depth map | 3-layer CNN | Feature map |
| Fusion | SNN attention × CNN feature (elementwise) | Smart Gate | Segmentation logits |

Ground truth: pixel-wise semantic segmentation masks. Task: urban driving scene understanding on DSEC only (nuScenes dropped).

---

## Repository & Data Paths

```
GitHub repo:   C:\Users\vvela\Documents\VT\AdvML\cnn-snn-lidar-fusion-comparison\
Dataset root:  C:\Users\vvela\Documents\VT\AdvML\dsec-data\
Conda env:     neural_arch  (Python 3.11)
```

---

## Confirmed Data Structure

```
dsec-data/
├── zurich_city_04_a/                          ← PRIMARY (fully processed in Session 1)
│   ├── zurich_city_04_a_calibration/
│   │   ├── cam_to_cam.yaml                    ← camera intrinsics + rectification
│   │   └── cam_to_lidar.yaml                  ← LiDAR-to-camera extrinsics ← SESSION 2 KEY FILE
│   ├── zurich_city_04_a_events_left/
│   │   ├── events.h5                          ← 358M DVS events (x,y,t,p)
│   │   └── rectify_map.h5
│   ├── zurich_city_04_a_disparity_event/      ← 351 disparity PNGs
│   ├── zurich_city_04_a_images_rectified_left/ ← RGB frames
│   ├── zurich_city_04_a_semantic/             ← segmentation GT labels
│   ├── zurich_city_04_a_disparity_timestamps.txt
│   ├── lidar_imu.bag                          ← ROS1 bag (~3.1 GB), 630 s recording
│   ├── lidar_sweeps/                          ← ✅ PRODUCED IN SESSION 1 (347 .npy files)
│   └── overlap_bounds_04a.npy                 ← [overlap_start, overlap_end] in µs
│
└── zurich_city_00_a/                          ← BACKUP — NOT NEEDED (pair count is sufficient)
```

---

## Session 1 — What Was Done

### Step 1 — Bag Inspection
- Library installed: `rosbags` (version ≥ 0.9), `hdf5plugin`, `h5py`, `pyyaml`, `opencv-python`
- Bag contains exactly 2 topics:
  - `/velodyne_points` → `sensor_msgs/msg/PointCloud2`  ← **the LiDAR topic**
  - `/imu/data` → `sensor_msgs/msg/Imu`  (not used)
- Bag duration: 630.2 s total (vehicle sitting + driving; most is outside the event window)
- Message count: 634,582 (dominated by IMU at ~1000 Hz)

### Step 2 — Timestamp Extraction & Overlap
Key discovery: **DSEC `events.h5` uses BLOSC compression and relative timestamps.**
- Must `import hdf5plugin` before opening `events.h5`, or h5py raises `OSError: Can't open directory`
- Absolute event timestamp = `events/t` (relative µs) + `t_offset` (scalar at root of file)
- `t_offset = 36,470,599,656 µs`

| Sensor | Time range (µs) | Duration |
|--------|-----------------|----------|
| LiDAR sweeps | 36,338,807,440 → 36,966,156,107 | 627.3 s |
| Event camera | 36,470,599,656 → 36,505,601,655 | 35.0 s |
| **Overlap window** | **36,470,599,656 → 36,505,601,655** | **35.0 s** |

- LiDAR sweep rate: 9.9 Hz
- LiDAR sweeps within overlap window: 347

### Step 3 — Point Cloud Extraction
- `rosbags ≥ 0.9` broke the old `deserialize_cdr` API. Correct replacement:
  ```python
  from rosbags.typesys import Stores, get_typestore
  typestore = get_typestore(Stores.ROS1_NOETIC)
  msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
  ```
- 347 `.npy` files saved to `dsec-data/zurich_city_04_a/lidar_sweeps/`
- Each file: `{timestamp_us:016d}.npy`, shape `(N, 4)` float32 → columns: x, y, z, intensity
- 5,874 sweeps skipped (outside overlap window, as expected)
- 0 errors

Sample sweep stats (mid-sequence):
- Points per sweep: ~24,952
- X range: −67.8 to +104.6 m
- Y range: −49.3 to +47.7 m
- Z range: −2.9 to +14.0 m
- Intensity: 0–209 (reflectance, unitless)

### Step 4 — Temporal Matching
- Frame timestamps loaded from `zurich_city_04_a_disparity_timestamps.txt`
- 351 semantic frames at 10.0 Hz covering the full 35 s overlap window
- Each frame nearest-neighbour matched to closest LiDAR sweep timestamp
- Filter: temporal gap < 55 ms (one sweep period + margin)
- 1 boundary pair dropped (last frame's nearest sweep was 28 ms past `overlap_end`, file not on disk)

**Final result: 350 valid matched pairs**

| Stat | Value |
|------|-------|
| Total frames | 351 |
| Valid pairs | **350** |
| Rejected | 1 (boundary) |
| Mean temporal gap | 25.19 ms |
| Max temporal gap | 50.33 ms |
| Median temporal gap | 25.00 ms |
| Unique LiDAR sweeps used | 347 (3 sweeps each serve 2 frames) |

Output JSON saved to repo: `data/zurich_city_04_a_pairs.json`

---

## Artifacts Produced

| File | Location | Description |
|------|----------|-------------|
| `03_lidar_extraction_alignment.ipynb` | `notebooks/` | Session 1 notebook |
| `zurich_city_04_a_pairs.json` | `data/` | 350 frame↔sweep pairs |
| `lidar_sweeps/*.npy` | `dsec-data/zurich_city_04_a/lidar_sweeps/` | 347 point clouds (**local only, not in git**) |
| `overlap_bounds_04a.npy` | `dsec-data/zurich_city_04_a/` | Overlap window bounds (**local only**) |
| `.gitignore` updated | repo root | Added `**/lidar_sweeps/`, `*.npy` exclusions |

---

## pairs JSON Schema

```json
{
  "frame_idx": 0,
  "frame_ts":  36470600624,
  "lidar_ts":  36470631369,
  "dt_ms":     30.745,
  "lidar_file": "0000036470631369.npy"
}
```

- `frame_idx`: index into `disparity_timestamps.txt` (0-based); also indexes into semantic label PNGs
- `frame_ts`: absolute timestamp of the semantic/disparity frame (µs)
- `lidar_ts`: absolute timestamp of the matched LiDAR sweep (µs)
- `dt_ms`: temporal gap between the two (always < 55 ms)
- `lidar_file`: filename of the `.npy` point cloud in `lidar_sweeps/`

---

## Known Quirks & Fixes (for future sessions)

| Issue | Root cause | Fix |
|-------|-----------|-----|
| `OSError: Can't open directory` on `events.h5` | BLOSC compression not loaded | `import hdf5plugin` before `import h5py` |
| `events/t` timestamps don't overlap with LiDAR | DSEC stores relative timestamps | Add `t_offset` from root of `events.h5` |
| `ImportError: deserialize_cdr` | rosbags ≥ 0.9 removed this function | Use `typestore.deserialize_ros1(rawdata, msgtype)` |
| 350 pairs ≠ 351 frames | Boundary sweep was outside overlap window | Drop pair, 350 is sufficient |

---

## Session 2 — What To Build Next

**Notebook:** `notebooks/04_lidar_projection.ipynb`
**Goal:** Project each LiDAR `.npy` sweep into the rectified left camera image plane to produce a 2D sparse depth map — the input to the CNN branch.

### Step-by-step plan

**Step 1 — Load calibration**
- Read `cam_to_lidar.yaml`: gives 4×4 extrinsic transform T_lidar_to_cam (rotation + translation)
- Read `cam_to_cam.yaml`: gives camera intrinsic matrix K (fx, fy, cx, cy) and image size for the rectified left camera
- Print both and verify shapes before proceeding

**Step 2 — Project one sweep (dry run)**
For a single mid-sequence pair from `zurich_city_04_a_pairs.json`:
1. Load `.npy` → (N, 4) array
2. Filter to points with x > 0 (forward-facing half-space only)
3. Apply extrinsic: `P_cam = R @ P_lidar.T + t` → (3, N)
4. Filter to points with z_cam > 0 (in front of camera)
5. Project to pixel: `u = fx * X/Z + cx`, `v = fy * Y/Z + cy`
6. Filter to points within image bounds (0 ≤ u < W, 0 ≤ v < H)
7. Scatter into depth map array: `depth_map[v, u] = Z`
8. Visualise with matplotlib (overlay on RGB frame to verify alignment)

**Step 3 — Batch projection**
- Run projection for all 350 pairs
- Save each depth map as `{frame_idx:04d}.npy` (float32, shape H×W, 0 = no return)
- Save to `dsec-data/zurich_city_04_a/depth_maps/` (exclude from git)

**Step 4 — Optional densification**
- Sparse depth maps have many zeros (LiDAR is sparse at range)
- Options: nearest-neighbour inpainting (`cv2.inpaint`), bilateral filtering, or leave sparse for now
- Decision: start sparse; densify only if CNN training loss is unstable

**Step 5 — Dataset class (stub)**
- Write a PyTorch `Dataset` that returns `(depth_map, semantic_label, time_surface)` for each `frame_idx`
- Depth map from `depth_maps/{frame_idx:04d}.npy`
- Semantic label from `zurich_city_04_a_semantic/` (PNG, indexed colour → class int)
- Time surface from Notebook 01 outputs
- Save to `src/datasets/dsec_dataset.py`

### Critical calibration note
DSEC's `cam_to_lidar.yaml` stores the transform as **LiDAR-in-camera-frame**, i.e. it maps LiDAR points INTO camera coordinates directly. Double-check the convention by projecting one known ground-return point and confirming it lands at the bottom of the image. If it lands at the top, invert the transform.

---

## Do NOT Do in Session 2
- Do not start SNN or CNN model training
- Do not process `zurich_city_00_a`
- Do not commit `depth_maps/` or `lidar_sweeps/` to git
- Do not attempt fusion or evaluation yet

---

## Completed Notebook Inventory

| Notebook | Status | Content |
|----------|--------|---------|
| `01_time_surface_dsec.ipynb` | ✅ complete | DVS event → time surface generation |
| `02_disparity_eda.ipynb` | ✅ complete | Disparity map EDA |
| `03_lidar_extraction_alignment.ipynb` | ✅ complete | LiDAR extraction + temporal matching |
| `04_lidar_projection.ipynb` | 🔲 Session 2 | LiDAR → camera projection → depth maps |
