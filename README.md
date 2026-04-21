# SNN vs CNN for LiDAR-Event Camera Fusion

**Course:** ECE 5424 Advanced Machine Learning | Virginia Tech, Spring 2026  
**Team:** Victor Velasquez · Michael Volkman · Enrique Maldonado · Christopher Quispesivana  
**Repository:** `cnn-snn-lidar-fusion-comparison`

---

## Overview

End-to-end comparison of Spiking Neural Networks (SNNs) and Convolutional Neural Networks (CNNs) for fusing event-camera and LiDAR data in autonomous driving perception.

Current sensor fusion pipelines force asynchronous, event-driven data into synchronous frames, which causes a mismatch that inflates latency and wastes computation. This project asks whether SNNs, which natively handle asynchronous spike-based data, can outperform CNNs at the fusion stage for urban driving scene understanding.

We build a two-branch fusion architecture on the DSEC dataset. Branch A processes real DVS event streams through a Time Surface representation and a 3-layer LIF SNN. Branch B processes projected LiDAR depth maps through a 3-layer CNN. A Smart Gate fuses both branches by using the SNN attention map to selectively suppress static LiDAR features. Both branches are evaluated side-by-side under identical conditions on semantic segmentation.

---

## Target Specifications

| Metric | Target | Status |
|--------|--------|--------|
| Segmentation accuracy (mIoU) | Competitive with CNN baseline | In progress |
| Inference latency (GPU) | Lower than CNN at equivalent mIoU | In progress |
| Smart Gate sparsity ratio | >30% LiDAR feature suppression | In progress |
| SNN training convergence | Surrogate gradient loss < CNN baseline loss | In progress |

---

## Engineering Decisions and Key Trade-offs

### 1. DSEC Only (nuScenes Dropped)
The original proposal called for fusing DSEC event data with nuScenes LiDAR. After analysis, DSEC already includes LiDAR data in rosbag format alongside the event camera, calibrated to the same vehicle coordinate frame. Using a single dataset eliminates the dataset merging problem entirely and gives us spatially and temporally consistent sensor data. nuScenes is noted as a natural extension for future work.

### 2. Real DVS Events (v2e Dropped)
DSEC provides real Dynamic Vision Sensor recordings stored in HDF5 format. Synthetic event generation with v2e is not needed. v2e remains available in the repository for potential use with other video datasets in future work.

### 3. Temporal Alignment as the Core Data Engineering Problem
The event camera and LiDAR do not share a clock. DSEC stores event timestamps relative to recording start while the LiDAR bag uses absolute ROS time, creating an apparent offset of roughly 10 hours. A `t_offset` scalar stored in `events.h5` resolves this. Even after clock alignment, the two sensors never fire at exactly the same microsecond. A 55ms quality filter (half one LiDAR sweep period) rejects pairs where the scene has changed too much between readings. This produced 350 valid matched pairs from the 35-second overlap window.

### 4. Camera-Plane LiDAR Projection (Not BEV)
Rather than a Bird's Eye View projection, we project LiDAR point clouds directly into the rectified left camera image plane. This puts the LiDAR depth map in the same coordinate frame as the event camera Time Surface, enabling pixel-level spatial alignment for the Smart Gate fusion. The Velodyne VLP-16 produces roughly 4,100 non-zero pixels per frame at 640x480 resolution.

### 5. Scaled Intrinsics for Direct Projection
DSEC calibrates the LiDAR to the frame camera at 1440x1080 native resolution. Projecting at native resolution and downsampling to 640x480 loses 78% of LiDAR points through nearest-neighbour collapse. Scaling the intrinsic matrix directly to 640x480 and projecting once recovers roughly 4x more depth returns per frame.

### 6. Time Surface as the Event Representation
Raw event streams are asynchronous lists of (x, y, t, polarity) tuples. A Time Surface converts this into a 2D grid where each pixel stores the timestamp of its most recent event, normalized within a 50ms window. Recently active pixels are bright, inactive pixels decay toward zero. This preserves temporal dynamics while producing a structured tensor both the SNN and CNN can process.

### 7. Smart Gate Fusion
Rather than concatenation or addition, the SNN attention map gates the CNN LiDAR features elementwise. Where the event stream detects motion, LiDAR features pass at full strength. Where the scene is static, LiDAR features are suppressed toward zero. This implements event-driven sparse computation at the feature map level and directly quantifies the computational savings of event-driven attention.

### 8. Controlled Baseline Design
The CNN baseline is architecturally identical to the SNN branch: same number of layers, same channel dimensions, same training data and optimizer. The only variable is LIF spiking neurons versus ReLU neurons. Any measured performance difference is attributable to the temporal integration behavior of the SNN, not architectural differences.

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
| SNN Branch A | 3-layer Leaky Integrate-and-Fire network trained with surrogate gradients via snnTorch; output (64, 120, 160) |
| Depth Map | LiDAR point cloud projected into camera plane using calibrated extrinsics; shape (1, 480, 640), units metres |
| CNN Branch B | 3-layer convolutional network with ReLU activations; identical structure to SNN branch; output (64, 120, 160) |
| Smart Gate | Elementwise multiplication of SNN attention map x CNN depth feature map |
| Segmentation Head | Upsamples fused features to (11, 480, 640); trained with CrossEntropyLoss against DSEC semantic labels |

---

## Dataset

**DSEC** (Driving Stereo Event Camera dataset) -- single dataset, no merging required.

| Property | Value |
|----------|-------|
| Sequence used | `zurich_city_04_a` (primary), `zurich_city_00_a` (backup) |
| Event camera | 640x480 DVS, real hardware recordings |
| LiDAR | Velodyne VLP-16, 16 beams, ~25k points/sweep |
| Semantic labels | 11 classes (road, vehicle, pedestrian, building, vegetation, etc.) |
| Training samples | 350 matched triplets (event window + LiDAR sweep + segmentation label) |
| Train / val split | 280 / 70 (80/20) |
| Temporal alignment | 55ms filter; mean gap 25.19ms; max gap 50.33ms |

---

## Evaluation Protocol

| Axis | Metric | Protocol |
|------|--------|----------|
| Accuracy | mIoU (mean Intersection over Union) | Computed per class and averaged across all 11 classes |
| Speed | End-to-end inference latency (ms) | GPU; averaged over 100 runs |
| Efficiency | Smart Gate sparsity ratio (%) | Fraction of LiDAR features suppressed per scene |

**Decision criterion:** SNN is considered superior if it matches CNN mIoU (within 2%) at lower latency, or achieves higher mIoU at equivalent speed.

---

## Notebook Pipeline

| Notebook | Status | Content |
|----------|--------|---------|
| `01_time_surface_dsec.ipynb` | Done | DVS events to Time Surface generation and visualization |
| `02_disparity_eda.ipynb` | Done | Disparity map exploratory data analysis |
| `03_lidar_extraction_alignment.ipynb` | Done | Rosbag extraction and temporal matching (350 pairs) |
| `04_lidar_projection.ipynb` | Done | LiDAR to camera projection and depth map generation |
| `05_snn_branch.ipynb` | In progress | Time surface loading, LIF SNN encoder, feature map output |
| `06_cnn_branch.ipynb` | In progress | CNN depth encoder, same output shape as SNN branch |
| `07_fusion_training.ipynb` | Pending | Smart Gate fusion, training loop, mIoU and latency evaluation |

---

## Tech Stack

| Category | Tool | Notes |
|----------|------|-------|
| Language | Python 3.11 | |
| Deep Learning | PyTorch 2.5.1 + CUDA 12.1 | GPU-accelerated training |
| SNN Framework | snnTorch | Surrogate gradient training for LIF neurons |
| Data | rosbags | ROS1 bag reading without ROS installation |
| Datasets | DSEC | Single dataset covering event camera, LiDAR, and semantic labels |
| Dev Environment | Jupyter + conda (neural_arch env) | |
| Version Control | Git + GitHub | |

---

## Results

*In progress -- results will be updated as training completes.*

| Model | mIoU | GPU Latency (ms) | Sparsity Ratio |
|-------|------|-----------------|----------------|
| CNN Baseline | -- | -- | N/A |
| SNN + Smart Gate | -- | -- | -- |

---

## Known Implementation Notes

| Issue | Fix |
|-------|-----|
| `events.h5` uses BLOSC compression | `import hdf5plugin` before `import h5py` |
| Event timestamps are relative, LiDAR timestamps are absolute | Add `t_offset` from root of `events.h5` |
| `cam_to_lidar.yaml` stores a 4x4 matrix, not separate R and t | Invert the full matrix: `R = R_lc.T`, `t = -R_lc.T @ t_lc` |
| Projecting at 1440x1080 then resizing loses 78% of LiDAR points | Scale intrinsics to 640x480, project directly |
| Semantic label PNGs are 640x440, depth maps are 640x480 | `cv2.resize(..., INTER_NEAREST)` in dataset class |
| Labels are 2 subdirectories deep | Path: `*_semantic/zurich_city_04_a/11classes/` |

---

## References

1. Y. Hu et al., "v2e: From Video Frames to Realistic DVS Events," CVPR Workshop, 2021.
2. J. Greene et al., "SENPI: A PyTorch-Enabled Tool for Synthetic Event Camera Data," SPIE, 2025.
3. A. Sironi et al., "HATS: Histograms of Averaged Time Surfaces for Event Cameras," CVPR, 2018.
4. T. Ali et al., "An FPGA-based Neuromorphic Vision System Accelerator," SPIE, 2024.
5. M. Isik et al., "Accelerating Sensor Fusion in Neuromorphic Computing: A Case Study on Loihi-2," arXiv:2408.16096, 2024.
6. J. Eshraghian et al., "Training Spiking Neural Networks Using Lessons From Deep Learning," Proc. IEEE, vol. 111, 2023.
7. M. Gehrig et al., "DSEC: A Stereo Event Camera Dataset for Driving Scenarios," IEEE RA-L, 2021.

---

*ECE 5424 Advanced Machine Learning -- Virginia Tech, Spring 2026*
