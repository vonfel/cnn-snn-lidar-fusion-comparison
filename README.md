# SNN vs CNN for LiDAR-Event Camera Fusion

**Course:** ECE 5424 Advanced Machine Learning | Virginia Tech, Spring 2026  
**Team:** Victor Velasquez · Michael Volkman · Enrique Maldonado · Christopher Quispesivana  
**Repository:** `cnn-snn-lidar-fusion-comparison`

---

## Overview

End-to-end comparison of Spiking Neural Networks (SNNs) and Convolutional Neural Networks (CNNs) for fusing event-camera and LiDAR data in autonomous driving perception.

Current sensor fusion pipelines force asynchronous, event-driven data into synchronous frames, which causes a mismatch that inflates latency and wastes computation. This project asks whether SNN-based temporal attention can improve CNN segmentation of LiDAR depth maps compared to naive channel concatenation, using real DVS event data and real Velodyne LiDAR from the DSEC dataset.

We build a three-way ablation: a depth-only CNN baseline, an early fusion CNN that concatenates both modalities, and a Smart Gate model where an SNN processes the event stream and gates the CNN LiDAR features elementwise. All three models are evaluated under identical conditions on semantic segmentation using the same fixed validation set.

---

## Why Three Models Instead of Two Branches

The original proposal framed this as Branch A (SNN on events) vs Branch B (CNN on LiDAR), asking which branch produces better features for object detection. After reviewing the event-based computing literature, that framing was revised for two reasons.

First, a head-to-head comparison between branches processing different modalities is not a fair test of the SNN vs CNN question. Any accuracy difference could come from the data, not the architecture. Second, the literature shows that SNNs serve as upstream context providers for CNNs in practical pipelines, not as competitors. The downstream processing pipeline in event-based computing goes SNNs to CNN/Conv-SNN to temporal models, meaning SNN output feeds CNN processing, not replaces it.

The three-model ablation properly isolates what each component contributes. Model A establishes the LiDAR-alone lower bound. Model B establishes the naive fusion ceiling. Model C tests whether SNN-driven attention adds something beyond having both modalities concatenated. This is a clean scientific question with a defensible answer regardless of which direction the result goes.

---

## Engineering Decisions and Key Trade-offs

### 1. DSEC Only (nuScenes Dropped)
The original proposal called for fusing DSEC event data with nuScenes LiDAR. After analysis, DSEC already includes LiDAR in rosbag format alongside the event camera, calibrated to the same vehicle coordinate frame. Using a single dataset eliminates the dataset merging problem and gives us spatially and temporally consistent sensor data. nuScenes is noted as a natural extension for future work.

### 2. Real DVS Events (v2e Dropped)
DSEC provides real Dynamic Vision Sensor recordings stored in HDF5 format. Synthetic event generation with v2e is not needed. v2e remains in the repository for reference.

### 3. Temporal Alignment as the Core Data Engineering Problem
The event camera and LiDAR do not share a clock. DSEC stores event timestamps relative to recording start while the LiDAR bag uses absolute ROS time, creating an apparent offset of roughly 10 hours. A `t_offset` scalar stored in `events.h5` resolves this. Even after clock alignment, both sensors never fire at exactly the same microsecond. A 55ms quality filter (half one LiDAR sweep period) rejects pairs where the scene has changed too much between readings. This produced 350 valid matched pairs from the 35-second overlap window on `zurich_city_04_a`.

### 4. Camera-Plane LiDAR Projection (Not BEV)
Rather than Bird's Eye View projection, we project LiDAR point clouds directly into the rectified left camera image plane. This puts the depth map in the same coordinate frame as the event camera Time Surface, enabling pixel-level spatial alignment for the Smart Gate fusion. The Velodyne VLP-16 produces roughly 4,100 non-zero pixels per frame at 640x480 resolution.

### 5. Scaled Intrinsics for Direct Projection
DSEC calibrates the LiDAR to the frame camera at 1440x1080 native resolution. Projecting at native resolution and downsampling to 640x480 loses 78% of LiDAR points through nearest-neighbour collapse. Scaling the intrinsic matrix directly to 640x480 and projecting once recovers roughly 4x more depth returns per frame.

### 6. Time Surface as the Event Representation
Raw event streams are asynchronous lists of (x, y, t, polarity) tuples. A Time Surface converts this into a 2D grid where each pixel stores the timestamp of its most recent event, normalized within a 50ms window. Recently active pixels are bright, inactive pixels decay toward zero. This preserves temporal dynamics while producing a structured tensor the models can process.

### 7. Three-Way Ablation Instead of Two-Branch Comparison
The ablation design (depth-only / early fusion / SNN Smart Gate) follows the standard pattern used in the event-vision fusion literature (HALSIE, EIFNet). It isolates the contribution of each component and provides a fair comparison because Model B and Model C see identical inputs, differing only in how those inputs are combined.

### 8. Depth Normalization
Raw LiDAR depth values range 0 to 104m while the time surface is normalized to [0,1]. This scale mismatch creates an elongated loss landscape that slows training. Dividing depth by 104.0 before training puts both modalities on the same scale and improved mIoU for Models A and B in Round 1.

### 9. Label Smoothing for Model C
Weighted CrossEntropyLoss is incompatible with the SNN branch. The SNN unrolls across 4 time steps, creating a deeper backward pass that amplifies gradients under class-weighted loss, causing repeated val_loss spikes regardless of weight configuration. Label smoothing (0.1) provides gentler regularization without the gradient instability.

---

## Architecture

```
                    +-----------------------------------------+
                    |           Branch A -- Events            |
                    |                                         |
  events.h5 ──────► |  Time Surface ──────────────────► SNN   |──► Attention Map
                    |  (2-channel, 480x640)          (3 LIF)  |         |
                    +-----------------------------------------+         |
                                                                   Smart Gate
                                                                  (elementwise x)
                    +-----------------------------------------+         |
                    |           Branch B -- LiDAR             |         |
                    |                                         |         |
  lidar_imu.bag ──► |  Depth Map ─────────────────────► CNN   |──► Feature Map
                    |  (projected, 480x640)          (3 Conv)  |         |
                    +-----------------------------------------+         |
                                                                         v
                                                                  Fused Feature Map
                                                                         |
                                                                         v
                                                               Segmentation Head
                                                                         |
                                                                         v
                                                          Semantic Labels (11 classes)
```

| Block | Description |
|-------|-------------|
| Time Surface | 2-channel (ON/OFF polarity) decay map built from a 50ms event window; shape (2, 480, 640) |
| SNN Branch A | 3-layer Leaky Integrate-and-Fire network trained with surrogate gradients via snnTorch; output (64, 120, 160). No MaxPool -- stride-2 convolutions only |
| Depth Map | LiDAR point cloud projected into camera plane using calibrated extrinsics; shape (1, 480, 640), normalized to [0,1] |
| CNN Branch B | 3-layer convolutional network with ReLU activations; identical structure to SNN branch; output (64, 120, 160) |
| Smart Gate | Elementwise multiplication of sigmoid(SNN attention map) x CNN depth feature map |
| Segmentation Head | Two ConvTranspose layers upsampling fused features to (11, 480, 640); CrossEntropyLoss against DSEC 11-class semantic labels |

---

## Three-Way Ablation Models

| Model | Class | Inputs | Params | Purpose |
|-------|-------|--------|--------|---------|
| A | `DepthOnlyCNN` | depth (1ch) | 61,963 | LiDAR-alone lower bound |
| B | `EarlyFusionCNN` | depth + time surface concat (3ch) | 62,251 | Naive multimodal fusion ceiling |
| C | `SmartGateModel` | depth + time surface separate branches | 85,403 | SNN attention mechanism |

Models A and B share the same `CNNEncoder` backbone. Model C adds an `SNNEncoder` branch whose mean firing rate serves as the spatial attention gate on CNN depth features.

---

## Dataset

**DSEC** (Driving Stereo Event Camera dataset) -- single dataset, no merging required.

| Property | Value |
|----------|-------|
| Primary sequence | `zurich_city_04_a` (highway, 35s, 350 matched pairs) |
| Secondary sequence | `zurich_city_00_a` (city street, 469 matched pairs, added in Round 2) |
| Event camera | 640x480 DVS, real hardware recordings, BLOSC-compressed HDF5 |
| LiDAR | Velodyne VLP-16, 16 beams, ~25k points/sweep, rosbag format |
| Semantic labels | 11 classes, ESS protocol (Cityscapes-aligned) |
| Combined training samples | 749 (280 from 04a + 469 from 00a) |
| Validation samples | 70 from 04a only, fixed across all rounds, seed=42 |
| Temporal alignment | 55ms filter; mean gap 25.19ms; max gap 50.33ms |

---

## Results

All models trained on combined dataset (749 samples), validated on `zurich_city_04_a` fixed split (70 samples).

| Model | Baseline | Round 1 | Round 2 | Round 3 |
|-------|----------|---------|---------|---------|
| A: Depth-only CNN | 0.0799 | 0.0871 | 0.0831 | in progress |
| B: Early Fusion CNN | 0.1457 | 0.1470 | 0.1385 | in progress |
| C: SNN + Smart Gate | 0.1284 | 0.1098 | 0.1035 | in progress |

Round configurations:
- **Baseline:** unweighted CE, raw depth (0-104m), 30+20 epochs
- **Round 1:** depth normalization, weighted CE, gradient clipping, 30 epochs
- **Round 2:** added zurich_city_00_a, EDA-informed weights, label smoothing for C, 30 epochs
- **Round 3:** AdamW + ReduceLROnPlateau, frequency-threshold weights (1% cutoff), 40 epochs -- in progress

---

## Key Scientific Findings

**C does not beat B across any round.** The SNN Smart Gate architecture does not outperform early concatenation fusion at this dataset scale (280-749 samples). This is the central finding and the honest scientific conclusion.

**Weighted CE is incompatible with SNN gradient unrolling.** Four time steps create a deeper backward pass that amplifies gradients under class-weighted loss, causing val_loss spikes of 5-9x regardless of weight configuration. This is a documented challenge in SNN training and motivates neuromorphic hardware deployment where backpropagation is eliminated entirely.

**EDA must precede loss function design.** Person (0.12%) and pole (1.25%) produce zero IoU across all rounds and all models despite progressive upweighting. A class frequency audit in the first notebook would have revealed this before any training.

**Vegetation: depth-only consistently beats fusion.** Model A outperforms B and C on vegetation in every round. Trees have distinctive LiDAR return profiles and the event camera adds noise for this class, not signal.

**Adding more data introduced distribution shift.** Round 2 regressed vs Round 1 for all models because training on 04a+00a mixed distribution while validating on 04a-only penalizes models that learned 00a-specific patterns. Cross-sequence training requires a matched cross-sequence validation set.

---

## Notebook Pipeline

| Notebook | Status | Content |
|----------|--------|---------|
| `01_time_surface_dsec.ipynb` | Done | DVS event to Time Surface, event camera EDA |
| `02_disparity_eda.ipynb` | Done | Disparity map exploratory data analysis |
| `03_lidar_extraction_alignment.ipynb` | Done | Rosbag extraction and temporal matching (350 pairs, 04a) |
| `04_lidar_projection.ipynb` | Done | LiDAR to camera projection, scaled intrinsics, depth maps |
| `05_models_and_data.ipynb` | Done | All three model architectures, real time surface integration |
| `06_training_and_eval.ipynb` | Done | Baseline training and evaluation |
| `06_training_and_eval_r1.ipynb` | Done | Round 1: depth normalization, weighted CE, per-class IoU |
| `07_r2_r3_all_combined.ipynb` | In progress | Round 2 and Round 3: combined dataset, AdamW, LR scheduler |

---

## Tech Stack

| Category | Tool | Notes |
|----------|------|-------|
| Language | Python 3.11 | |
| Deep Learning | PyTorch 2.5.1 + CUDA 12.1 | GPU-accelerated training |
| SNN Framework | snnTorch 0.9.4 | Surrogate gradient (fast sigmoid) for LIF neurons |
| Data | rosbags | ROS1 bag reading without ROS installation |
| Datasets | DSEC | Single dataset covering event camera, LiDAR, and semantic labels |
| Dev Environment | Jupyter + conda (neural_arch env) | |
| Version Control | Git + GitHub | |

---

## Known Implementation Notes

| Issue | Fix |
|-------|-----|
| `events.h5` uses BLOSC compression | `import hdf5plugin` before `import h5py` |
| Event timestamps are relative, LiDAR timestamps are absolute | Add `t_offset` from root of `events.h5` to get absolute µs |
| `cam_to_lidar.yaml` stores a 4x4 matrix, not separate R and t | Invert the full matrix: `R = R_lc.T`, `t = -R_lc.T @ t_lc` |
| Projecting at 1440x1080 then resizing loses 78% of LiDAR points | Scale intrinsics to 640x480, project directly |
| Semantic label PNGs are 640x440, depth maps are 640x480 | `cv2.resize(..., INTER_NEAREST)` in dataset class |
| Labels are 2 subdirectories deep | Path: `*_semantic/zurich_city_04_a/11classes/` |
| MaxPool breaks SNN training | Stride-2 convolutions only throughout SNN branch |
| Weighted CE causes gradient spikes in Model C | Use `label_smoothing=0.1` for Model C |
| Frame 0 has near-zero event sparsity | Car stationary at sequence start, normal behavior |
| 00a depth range slightly exceeds 104m norm cap | Values clip to ~1.04, negligible effect |
| `torch.load` FutureWarning | Add `weights_only=True` to all checkpoint loads |

---

## References

1. Y. Hu et al., "v2e: From Video Frames to Realistic DVS Events," CVPR Workshop, 2021.
2. J. Greene et al., "SENPI: A PyTorch-Enabled Tool for Synthetic Event Camera Data," SPIE, 2025.
3. A. Sironi et al., "HATS: Histograms of Averaged Time Surfaces for Event Cameras," CVPR, 2018.
4. T. Ali et al., "An FPGA-based Neuromorphic Vision System Accelerator," SPIE, 2024.
5. M. Isik et al., "Accelerating Sensor Fusion in Neuromorphic Computing: A Case Study on Loihi-2," arXiv:2408.16096, 2024.
6. J. Eshraghian et al., "Training Spiking Neural Networks Using Lessons From Deep Learning," Proc. IEEE, vol. 111, 2023.
7. M. Gehrig et al., "DSEC: A Stereo Event Camera Dataset for Driving Scenarios," IEEE RA-L, 2021.
8. S. Biswas et al., "HALSIE: Hybrid Approach to Learning Segmentation by Simultaneously Exploiting Image and Event Modalities," WACV, 2024.
9. Z. Sun et al., "ESS: Learning Event-based Semantic Segmentation from Still Images," ECCV, 2022.
10. R. Gaurav et al., "Spiking Approximations of the MaxPooling Operation in Deep SNNs," IJCNN, 2022.

---

*ECE 5424 Advanced Machine Learning -- Virginia Tech, Spring 2026*
