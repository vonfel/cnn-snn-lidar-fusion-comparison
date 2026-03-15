# SNN vs CNN for LiDAR-Event Camera Fusion : Self-Driving Perception

**Course:** ECE 5424 Advanced Machine Learning | Virginia Tech, Spring 2026  
**Team:** Enrique Maldonado · Christopher Quispesivana · Victor Velasquez · Michael Volkman  
**Repository:** `cnn-snn-lidar-fusion-comparison`

---

## Overview

End-to-end comparison of **Spiking Neural Networks (SNNs)** and **Convolutional Neural Networks (CNNs)** for fusing event-camera and LiDAR data in autonomous driving perception.

Current sensor fusion pipelines force asynchronous, event-driven data into synchronous frames — a mismatch that inflates latency and wastes computation. This project asks a direct question: **can SNNs, which natively handle asynchronous spike-based data, outperform CNNs at the fusion stage?**

We build a two-branch fusion architecture with a **Smart Gate** — an attention mechanism that uses the SNN's event-driven output to selectively suppress static LiDAR features, reducing computation on regions with no detected motion. Both models are evaluated side-by-side under identical conditions on DSEC and nuScenes.

---

## Target Specifications

| Metric | Target | Status |
|--------|--------|--------|
| Detection accuracy (mAP @ IoU 0.5) | Competitive with CNN baseline | In progress |
| Detection accuracy (mAP @ IoU 0.7) | Within ±2% of CNN baseline | In progress |
| Inference latency (GPU) | Lower than CNN at equivalent mAP | In progress |
| Smart Gate sparsity ratio | >30% LiDAR feature suppression on highway scenes | In progress |
| SNN training convergence | Surrogate gradient loss < CNN baseline loss | In progress |

---

## Engineering Decisions & Key Trade-offs

### 1. v2e Over SENPI for Synthetic Event Generation
Physical event cameras (DVS/DAVIS) are specialized hardware costing thousands of dollars. We use **v2e** (`sensorsINI/v2e`) to synthesize realistic event streams from standard driving videos. v2e was selected over SENPI due to its larger user base, active maintenance, published validation on real DVS hardware, and a well-documented tutorial notebook. SENPI is newer and less battle-tested for this use case.

### 2. Time Surface as the Bridge Representation
Raw event streams are sparse and asynchronous — neural networks expect dense, structured tensors. A **Time Surface** maps each pixel to the timestamp of its most recent event, producing a 2D image where recently-active pixels are bright and inactive pixels decay toward zero. This converts the event stream into a format CNNs and SNNs can both process, while preserving temporal dynamics that standard frame-based cameras discard.

### 3. Bird's Eye View (BEV) Projection for LiDAR
Raw LiDAR point clouds are unstructured 3D data. Projecting downward into a **Bird's Eye View** grid converts the point cloud into a regular 2D tensor — fully compatible with standard CNN convolutions — while preserving the spatial geometry critical for 3D bounding box detection.

### 4. Smart Gate Fusion (Element-wise Attention)
The most architecturally novel component. Rather than a standard concatenation or addition fusion, we use the SNN's Time Surface attention map as a **multiplicative gate** on the LiDAR CNN features. Where the event stream detects motion (bright Time Surface pixels), LiDAR features pass at full strength. Where the scene is static, LiDAR features are suppressed toward zero. This implements event-driven sparse computation at the feature-map level — the same principle that makes neuromorphic hardware efficient.

### 5. Controlled Baseline Design
To isolate the effect of neuron type, the CNN baseline is architecturally identical to the SNN branch: same number of layers, same channel dimensions, same training data and optimizer. The only variable is LIF spiking neurons vs. ReLU neurons. Any measured performance gap is therefore attributable to the SNN's temporal integration behavior, not to an unfair architectural advantage.

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              Branch A — Events           │
                    │                                         │
  Driving Video ──► │  v2e Simulator ──► Time Surface ──► SNN │──► Attention Map
                    │                    (2D decay map)  (LIF) │         │
                    └─────────────────────────────────────────┘         │
                                                                   Smart Gate
                                                                  (element-wise ×)
                    ┌─────────────────────────────────────────┐         │
                    │              Branch B — LiDAR            │         │
                    │                                         │         │
  LiDAR Scan ─────► │  Point Cloud ──► BEV Projection ──► CNN │──► Feature Map
                    │  (300K pts)      (2D top-down grid) (ReLU)│         │
                    └─────────────────────────────────────────┘         │
                                                                         ▼
                                                                  Fused Feature Map
                                                                         │
                                                                         ▼
                                                               Detection Head (FC layers)
                                                                         │
                                                                         ▼
                                                          3D Bounding Boxes (x, y, z, l, w, h, conf)
```

| Block | Description |
|-------|-------------|
| v2e Simulator | Converts standard video frames to realistic DVS event streams `(x, y, t, polarity)` |
| Time Surface | Maps each pixel to timestamp of most recent event; decays exponentially with time constant τ |
| SNN (Branch A) | 3-layer Leaky Integrate-and-Fire network; trained with surrogate gradients via snnTorch |
| BEV Projection | Projects LiDAR point cloud to 2D top-down grid with height and intensity channels |
| CNN (Branch A Baseline / Branch B) | 3-layer convolutional network; ReLU activations; identical structure to SNN branch |
| Smart Gate | Element-wise multiplication of SNN attention map × CNN LiDAR feature map |
| Detection Head | Fully-connected layers outputting 3D bounding box predictions with confidence scores |

---

## Evaluation Protocol

Three axes, following the nuScenes benchmark standard:

| Axis | Metric | Protocol |
|------|--------|----------|
| **Accuracy** | mAP @ IoU 0.5 and IoU 0.7 | nuScenes benchmark protocol for cross-paper comparability |
| **Speed** | End-to-end inference latency (ms) | CPU and GPU; averaged over 100 runs to account for variance |
| **Efficiency** | Smart Gate sparsity ratio (%) | % of LiDAR features suppressed per scene; reported across scene types (highway, intersection, parking) |

**Decision criterion:** SNN is considered superior if it matches CNN accuracy (within ±2% mAP) at lower latency, or achieves higher mAP at equivalent speed.

---

## Repository Structure

```
snn-lidar-fusion/
├── README.md
├── .gitignore
├── requirements.txt
├── environment.yml
├── src/
│   ├── data/
│   │   ├── v2e_pipeline.py          ← Synthetic event generation from video
│   │   ├── time_surface.py          ← Time Surface construction from event stream
│   │   ├── lidar_bev.py             ← LiDAR point cloud → BEV projection
│   │   └── dsec_loader.py           ← DSEC dataset loader
│   ├── models/
│   │   ├── snn_branch.py            ← 3-layer LIF SNN (snnTorch)
│   │   ├── cnn_branch.py            ← 3-layer CNN baseline
│   │   ├── smart_gate.py            ← Element-wise attention fusion
│   │   └── detection_head.py        ← 3D bounding box prediction layers
│   ├── train.py                     ← Training loop (SNN + CNN)
│   └── evaluate.py                  ← mAP, latency, sparsity benchmarks
├── notebooks/
│   ├── 01_v2e_setup_demo.ipynb      ← v2e installation and first output
│   ├── 02_time_surface_viz.ipynb    ← Time Surface construction + visualization
│   ├── 03_lidar_bev_viz.ipynb       ← BEV projection visualization
│   └── 04_fusion_pipeline.ipynb     ← End-to-end pipeline demo
├── docs/
│   ├── milestone_report.pdf
│   ├── final_report.pdf
│   └── figures/
│       ├── architecture_diagram.png
│       ├── time_surface_example.png
│       └── bev_example.png
├── results/
│   ├── snn_map_results.csv
│   ├── cnn_map_results.csv
│   ├── latency_benchmarks.csv
│   └── sparsity_analysis.csv
└── data/
    └── README.md                    ← Instructions to download DSEC + nuScenes
```

---

## Tech Stack

| Category | Tool | Notes |
|----------|------|-------|
| **Language** | Python 3.10+ | |
| **Deep Learning** | PyTorch 2.x | SNN and CNN training |
| **SNN Framework** | snnTorch | Surrogate gradient training for LIF neurons |
| **Event Synthesis** | v2e (`sensorsINI/v2e`) | DVS event stream generation from video |
| **Datasets** | DSEC, nuScenes | Primary evaluation benchmarks |
| **Dev Environment** | Conda + Jupyter | `environment.yml` provided |
| **Version Control** | Git + GitHub | This repository |
| **Compute** | Personal GPUs / TBD | Training and inference benchmarking |

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/vonfel/snn-lidar-fusion.git
cd snn-lidar-fusion

# Create and activate the conda environment
conda env create -f environment.yml
conda activate snn-fusion

# Install v2e for event synthesis
pip install v2e

# Download DSEC dataset
# See data/README.md for download instructions

# Run the Time Surface demo notebook
jupyter notebook notebooks/02_time_surface_viz.ipynb
```

---

## Results

*In progress — results will be populated as experiments complete.*

| Model | mAP @ IoU 0.5 | mAP @ IoU 0.7 | GPU Latency (ms) | Sparsity Ratio |
|-------|--------------|--------------|-----------------|----------------|
| CNN Baseline | — | — | — | N/A |
| SNN + Smart Gate | — | — | — | — |

---

## References

1. Y. Hu et al., "v2e: From Video Frames to Realistic DVS Events," *CVPR Workshop*, 2021.
2. J. Greene et al., "SENPI: A PyTorch-Enabled Tool for Synthetic Event Camera Data," *SPIE*, 2025.
3. A. Sironi et al., "HATS: Histograms of Averaged Time Surfaces for Event Cameras," *CVPR*, 2018.
4. T. Ali et al., "An FPGA-based Neuromorphic Vision System Accelerator," *SPIE*, 2024.
5. M. Isik et al., "Accelerating Sensor Fusion in Neuromorphic Computing: A Case Study on Loihi-2," *arXiv:2408.16096*, 2024.
6. J. Eshraghian et al., "Training Spiking Neural Networks Using Lessons From Deep Learning," *Proc. IEEE*, vol. 111, 2023.

---

*ECE 5424 Advanced Machine Learning — Virginia Tech, Spring 2026*
