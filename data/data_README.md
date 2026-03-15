# Dataset Download Instructions

## DSEC
53 real driving sequences with stereo event camera + LiDAR.

```bash
# Download from official source
# https://dsec.ifi.uzh.ch/dsec-datasets/download/
```

For development, download 2–3 sequences only. Full dataset is large.

## nuScenes
1,000 scenes, 1.4M annotated 3D bounding boxes. Used for mAP benchmarking.

```bash
# Register and download from
# https://www.nuscenes.org/nuscenes#download
# Mini split (10 scenes) is sufficient for development
```

## v2e Synthetic Events
Generated on-the-fly from DSEC driving video using v2e. No separate download required.
See `notebooks/01_v2e_setup_demo.ipynb` for generation instructions.
