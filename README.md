# Object-Detection

# Object Detection with MobileNet-SSD

A computer vision pipeline that detects and labels physical objects in
both **static images** and **video** using a pre-trained MobileNet-SSD
neural network. Built as an extension project alongside a DecodeLabs
AI internship — this explores Path 2 (Object Detection) rather than
Path 1 (OCR), which was completed separately.

## What This Does

Given an image or video, the model:
1. Locates every recognizable object in the frame
2. Draws a bounding box around each one
3. Labels it with a class name and confidence score

It recognizes 20 object classes (from the PASCAL VOC dataset):
person, car, bicycle, dog, cat, bus, train, aeroplane, bird, boat,
bottle, chair, cow, diningtable, horse, motorbike, pottedplant, sheep,
sofa, and tvmonitor.

## Why MobileNet-SSD

- **MobileNet** is the backbone architecture — it uses *depthwise
  separable convolutions* (filtering each color channel separately
  instead of all at once), making it dramatically lighter and faster
  than a standard CNN. Designed for real-time inference on modest
  hardware, not big servers.
- **SSD (Single Shot Detector)** looks at the image once and predicts
  every object and its location in a single forward pass — no
  repeated scanning across the image like older detection methods.

The model is **pre-trained** on PASCAL VOC — like almost all real-world
computer vision projects, this doesn't train a network from scratch,
it loads finished, published weights and runs inference on them.

## How It Works

Every detection — image or video — follows the same three-part
pipeline:

**1. Blob construction (pre-processing)**
```python
blob = cv2.dnn.blobFromImage(
    image, scalefactor=0.007843, size=(300, 300), mean=127.5,
)
```
The image is resized to 300×300, mean-centered, and scaled — matching
the exact numerical format the network expects. Skipping this step is
the most common reason a detection network returns garbage.

**2. Forward pass**
```python
net.setInput(blob)
detections = net.forward()
```
One pass through the network returns every candidate detection at once.

**3. Confidence filtering + drawing**
Each detection carries a confidence score — the network's own
statistical estimate of certainty, never treated as ground truth.
Detections below a threshold are discarded; everything else gets a
colored, labeled bounding box.

For video, this exact same three-step process just runs once per
frame, in a loop, fast enough to keep up with playback speed.

## Files

| File | Purpose |
|---|---|
| `detector.py` | Runs detection on a single static image |
| `detector_video.py` | Runs detection on every frame of a video |
| `MobileNetSSD_deploy.prototxt` | Network architecture definition |
| `MobileNetSSD_deploy.caffemodel` | Pre-trained weights (~23MB) |
| `sample_objects.jpg` | Test image (dog, bicycle, car) |
| `test_video_1.avi` | Test video (pedestrians in a parking lot) |
| `detected_output.png` | Example output from the image detector |
| `detected_video_output.mp4` | Example output from the video detector |

**Important:** the `.prototxt` and `.caffemodel` must come from the
same source/release. Mixing a prototxt and weights file from two
different repos will usually fail to load, even though both are
technically valid MobileNet-SSD files — the layer definitions won't
match up.

## Results

**Image test** (`sample_objects.jpg` — a dog next to a bicycle, truck
in the background):
```
bicycle       99.8%
car           99.4%
dog           96.7%
```

**Video test** (`test_video_1.avi` — 795 frames, pedestrians and cars
in a parking lot):
```
person       2260 detections
car          163 detections
aeroplane    82 detections
```
Processed at ~47 fps on an M-series/Intel Mac (CPU only, no GPU) —
fast enough to be considered near real-time.

## How to Run

```bash
pip install opencv-python numpy
```

**On an image:**
```bash
python3 detector.py sample_objects.jpg
```

**On a video:**
```bash
python3 detector_video.py test_video_1.avi
```

Both scripts save an annotated output file (`detected_output.png` or
`detected_video_output.mp4`) in the same folder and print a summary of
everything detected.

**Note on output video playback:** the `.mp4` output uses OpenCV's
`mp4v` codec, which some editors (including VS Code's built-in
previewer) can't render inline. If the preview shows an error, just
open the file directly in a standard video player (QuickTime, VLC,
etc.) — the file itself is valid. To get a more universally-compatible
file, re-encode it with ffmpeg:
```bash
ffmpeg -i detected_video_output.mp4 -vcodec libx264 -pix_fmt yuv420p detected_video_output_fixed.mp4
```

## A Note on Confidence Thresholds

This particular pre-trained model (an older, VOC-era MobileNet-SSD)
reports lower raw confidence scores than modern detectors, even on
completely correct detections. Rather than assuming a fixed threshold
from any single source and calling it done, both scripts use a lower,
explicitly-commented threshold — and on real test data, detections
still cleared 96%+ regardless. The takeaway: always check what a
specific model's confidence scores actually look like on real data
before deciding where to draw the cutoff.

## Key Concepts

- Object detection vs. classification — locating *and* labeling
  multiple things in one frame, not just categorizing a single input
- Blob construction and why pre-processing must exactly match the
  conditions the network was trained under
- Single Shot Detection (SSD) vs. older sliding-window detection
- Confidence scores as probability, not certainty
- Applying an image pipeline to video — a video is just a sequence of
  images processed one at a time, fast enough to keep up
- Using a pre-trained model rather than training from scratch — the
  standard approach for nearly all real-world computer vision work

## Possible Extensions

- Swap in a modern detector (YOLOv8 via `ultralytics`) for higher
  baseline confidence and more object classes
- Run on live webcam input instead of a static video file
- Track individual objects across frames (not just detect per-frame)
- Trigger custom logic when a specific class is detected (e.g., alert
  on "person" after hours)
