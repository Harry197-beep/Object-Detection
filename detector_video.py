"""
DecodeLabs - Project 4: Image or Text Recognition (Basic)
Batch 2026

Path 2 Extension: Object Detection on VIDEO (frame-by-frame)

Same MobileNet-SSD model and pipeline as detector.py, applied to every
frame of a video instead of a single static image. This is the natural
extension mentioned in the brief's "Emerging Horizons" slide - the same
IPO logic scales from a single image to a continuous stream of them.

Pipeline (IPO), repeated once per frame:
  INPUT   -> Read one frame from the video
  PROCESS -> Blob construction -> forward pass through the network
  OUTPUT  -> Draw boxes + labels on that frame, write it to the output video

Nothing about the model or the math changes - a video is just a
sequence of images (typically 24-30 per second), so "detecting objects
in a video" really means "detecting objects in an image, many times in
a row, fast enough to keep up."

Performance note:
MobileNet-SSD is specifically chosen here (over a heavier network)
because it's light enough to run per-frame on CPU without falling
hopelessly behind - this is exactly what "optimized for real-time
inference on edge devices" (from the Project 4 slides) is for.
"""

import sys
import time
import cv2
import numpy as np

PROTOTXT_PATH = "MobileNetSSD_deploy.prototxt"
MODEL_PATH = "MobileNetSSD_deploy.caffemodel"
VIDEO_PATH = "test_video.avi"
OUTPUT_VIDEO_PATH = "detected_video_output.mp4"

CONFIDENCE_THRESHOLD = 0.3  # see detector.py for why this isn't 0.8

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype="uint8")


def load_model(prototxt_path: str, model_path: str):
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    except cv2.error as e:
        raise FileNotFoundError(
            f"Could not load model files. Make sure '{prototxt_path}' and "
            f"'{model_path}' are in the same folder as this script."
        ) from e
    return net


def detect_frame(net, frame, confidence_threshold: float):
    """Run the same INPUT->PROCESS->OUTPUT pipeline on a single frame."""
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=0.007843, size=(300, 300), mean=127.5,
    )
    net.setInput(blob)
    detections = net.forward()

    counts = {}

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < confidence_threshold:
            continue

        class_id = int(detections[0, 0, i, 1])
        label_name = CLASSES[class_id] if class_id < len(CLASSES) else "unknown"
        counts[label_name] = counts.get(label_name, 0) + 1

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype("int")

        color = tuple(int(c) for c in COLORS[class_id])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{label_name}: {confidence * 100:.0f}%"
        label_y = y1 - 8 if y1 - 8 > 10 else y1 + 18
        cv2.putText(
            frame, label, (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )

    return frame, counts


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH

    print("Loading model (MobileNet-SSD)...")
    net = load_model(PROTOTXT_PATH, MODEL_PATH)

    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at '{video_path}'.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    print(f"Processing {total_frames} frames at {fps:.1f} fps...")
    start_time = time.time()
    frame_count = 0
    total_detections = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, counts = detect_frame(net, frame, CONFIDENCE_THRESHOLD)
        writer.write(annotated_frame)

        for label, n in counts.items():
            total_detections[label] = total_detections.get(label, 0) + n

        frame_count += 1
        if frame_count % 20 == 0 or frame_count == total_frames:
            elapsed = time.time() - start_time
            print(f"  Frame {frame_count}/{total_frames}  "
                  f"({frame_count / elapsed:.1f} fps processing speed)")

    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s ({frame_count / elapsed:.1f} fps average).")
    print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")

    print("\n=== Total detections across all frames ===")
    if not total_detections:
        print("(no objects detected above threshold)")
    else:
        for label, n in sorted(total_detections.items(), key=lambda x: -x[1]):
            print(f"{label:12s} {n} detections")


if __name__ == "__main__":
    main()