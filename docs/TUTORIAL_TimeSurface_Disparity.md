# Tutorial: Understanding Time Surface and Disparity Maps

This guide explains two core concepts in our pipeline - the **Time Surface** (Branch A)
and **Disparity Maps** (Branch B) - so every teammate can understand what the code
is doing and why, before touching any implementation.

---

## Branch A - Time Surface

### What is an Event Camera?

A standard camera captures full frames at a fixed rate (e.g., 30 fps). Every pixel
is read out simultaneously, whether or not it changed. This wastes bandwidth on
static regions and introduces motion blur at high speeds.

An **event camera** works completely differently. Each pixel fires **independently**
the moment it detects a brightness change, producing a continuous stream of events:

```
event = (x, y, timestamp, polarity)
```

- **(x, y)** - pixel location where the change occurred
- **timestamp** - microsecond-precision time of the change
- **polarity** - `+1` if the pixel got brighter, `-1` if it got darker

No motion = no events = zero wasted computation. A moving car edge generates
thousands of events per millisecond. An empty sky generates none.

### The Problem: SNNs Need Structured Input

Raw events are sparse and asynchronous - they arrive at irregular times and only
at pixels that changed. A neural network expects a regular 2D tensor as input.
We need to convert the event stream into a format the network can process while
preserving the temporal information that makes event cameras valuable.

### The Solution: Time Surface

A **Time Surface** asks a simple question for every pixel: **"how recently did
this pixel last fire?"**

For each pixel `(x, y)`, we store the timestamp of its most recent event and apply
exponential decay:

```
TS[y, x] = exp( -(t_current - t_last[y, x]) / τ )
```

| Symbol | Meaning |
|--------|---------|
| `t_current` | Timestamp of the most recent event in the window |
| `t_last[y, x]` | Timestamp of the last event at pixel (x, y) |
| `τ (tau)` | Time constant - controls how fast old events fade (we use 50 ms) |

**Intuition:**
- **Bright pixel (value ≈ 1.0)** → event just happened here. A pedestrian or car
  edge is actively moving through this pixel right now.
- **Dark pixel (value ≈ 0.0)** → nothing has changed here recently. Static
  background, road surface, parked objects.

The result is a regular 2D image - the **thermal footprint of recent motion**.
This is exactly what feeds our SNN branch: not a raw event stream, but a structured
map of *where the scene is currently active*.

### Why This Matters for Fusion

The Time Surface becomes our **attention signal**. In the Smart Gate fusion step,
it tells the LiDAR branch: *"these are the regions worth paying attention to."*
LiDAR features in bright Time Surface regions pass through at full strength.
LiDAR features in dark regions get suppressed. This is event-driven sparse
computation at the feature-map level.

### What We Used: DSEC Dataset

DSEC (`zurich_city_04_a`) contains real DVS sensor data from a vehicle-mounted
event camera in Zurich. The `events.h5` file stores millions of real hardware
events - no simulation needed. This gives us **ground-truth validation** of our
Time Surface implementation against actual sensor output.

**Notebook:** `notebooks/01_time_surface_dsec.ipynb`

---

## Branch B - Disparity Maps

### What is Disparity?

DSEC uses a **stereo event camera** - two cameras separated by a known baseline
distance (43.7 mm). When both cameras observe the same scene, a nearby object
appears at *different horizontal positions* in the left and right images. A distant
object appears at nearly the *same position* in both.

**Disparity** is the pixel distance between where the same point appears in the
left vs. right image.

```
depth (metres) = (focal_length_px × baseline_m) / disparity_px
```

| Value | Meaning |
|-------|---------|
| High disparity | Object is **close** - large pixel shift between cameras |
| Low disparity | Object is **far** - small pixel shift |
| Zero disparity | Background / sky - too far to measure reliably |

### DSEC Q8.8 Encoding

DSEC stores disparity as `uint16` values in **Q8.8 fixed-point format**. This means
the integer and fractional parts of the disparity value are packed into 16 bits,
with 8 bits for the integer part and 8 bits for the fractional part.

To convert to real disparity in pixels:

```python
disparity_pixels = raw_uint16 / 256.0
```

Then apply the depth formula with DSEC's calibration parameters:
- Focal length: `569.29 px`
- Baseline: `0.0437 m`

### Why Disparity for Now, LiDAR Later?

DSEC provides disparity maps (depth from stereo vision) rather than raw LiDAR
point clouds. These serve as our **initial depth modality exploration**.

For the full pipeline, Branch B will use **nuScenes LiDAR** - raw 3D point clouds
stored as `.bin` files - which are directly projected into a Bird's Eye View (BEV)
grid for the CNN branch. The disparity work here builds our intuition for depth
data before we move to true 3D point clouds.

### Bird's Eye View (BEV) - Coming Next

A LiDAR point cloud is an unstructured set of 3D points `(x, y, z, intensity)`.
A CNN needs a regular 2D tensor. The **Bird's Eye View projection** looks straight
down at the point cloud and bins points into a 2D grid - like a satellite photo
made of laser dots. This preserves the spatial geometry (car shapes, pedestrian
footprints) in a format the CNN can process with standard convolutions.

**Notebook:** `notebooks/02_disparity_eda.ipynb`

---

## How the Two Branches Connect

```
DSEC events.h5 ──► Time Surface ──► SNN (Branch A) ──► Attention Map ──┐
                                                                         │
                                                                    Smart Gate (×)
                                                                         │
nuScenes .bin ───► BEV Projection ──► CNN (Branch B) ──► Feature Map ───┘
                                                                         │
                                                              Detection Head
                                                                         │
                                                          3D Bounding Boxes
```

The Smart Gate is the key innovation: the SNN's attention map (which pixels had
recent motion) gates the CNN's LiDAR features (which regions have geometric
structure). Only regions where *both* motion and geometry are present contribute
to the final detection.

---
