"""Real-time fire and human detection using webcam / drone camera.

Controls:
- q: quit

Notes:
- Fire, smoke, and human detection now from a single YOLO model.
- For speed, YOLO inference is performed every N frames.
"""

import cv2
import math
import os
import time
import torch
from ultralytics import YOLO


# ============================================================
# Input source
# ============================================================
# Use 0 for laptop webcam.
# For ANAFI stream, we can replace this with camera source
INPUT_SOURCE = 0

# ============================================================
# Model paths
# ============================================================

VISION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.dirname(VISION_ROOT))
DETECTION_MODEL_PATH = os.path.join(PROJECT_ROOT, "firesmokehuman.pt")


# ============================================================
# YOLO inference parameters
# ============================================================

YOLO_CONFIDENCE = 0.40
YOLO_IOU = 0.50

# Remove overlapping detections for the same class after prediction.
DETECTION_NMS_IOU = 0.55

# Human-specific duplicate cleanup.
# Drop small human boxes that are almost fully inside another larger human box
# (e.g., hand/arm falsely detected as a separate person).
PERSON_NESTED_IOMIN = 0.85
PERSON_NESTED_AREA_RATIO = 0.45

# Smaller image size improves speed.
# 640 = better accuracy, slower.
# 416 or 320 = faster, lower accuracy.
YOLO_IMAGE_SIZE = 416

# Run YOLO every N frames.
# 1 = detect every frame, more accurate but slower.
# 2 or 3 = faster, slightly delayed detection.
PROCESS_EVERY_N_FRAMES = 3

# Resize incoming camera frame before processing.
FRAME_WIDTH = 640
FRAME_HEIGHT = 360

# Use GPU if available.
YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"
YOLO_HALF = torch.cuda.is_available()


# ============================================================
# Class names
# ============================================================

FIRE_CLASS_NAMES = ("fire", "flame")
SMOKE_CLASS_NAMES = ("smoke",)
PERSON_CLASS_NAMES = ("person", "human")
VALID_CLASS_NAMES = FIRE_CLASS_NAMES + SMOKE_CLASS_NAMES + PERSON_CLASS_NAMES


# ============================================================
# Tracking parameters
# ============================================================

TRACK_MAX_MISSES = 15
TRACK_MAX_MATCH_DISTANCE = 120
BBOX_EMA_ALPHA = 0.35
CONFIDENCE_EMA_ALPHA = 0.30
MIN_IOU_FOR_MATCH = 0.05

# Fire-person interaction.
INTERACTION_IOMIN_THRESHOLD = 0.15

# Temporal confirmation.
FIRE_CONFIRMATION_FRAMES = 5
PERSON_CONFIRMATION_FRAMES = 3


# ============================================================
# Colors and font
# ============================================================

GREEN = (0, 255, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_COMPLEX


# ============================================================
# Geometry utilities
# ============================================================

def bbox_center(box):
    x, y, w, h = box
    return (x + (w / 2.0), y + (h / 2.0))


def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def bbox_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union_area = max(1, area_a + area_b - inter_area)

    return inter_area / union_area


def bbox_intersection_area(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    return inter_w * inter_h


def bbox_iomin(box_a, box_b):
    """Intersection over the smaller box area.
    """
    inter = bbox_intersection_area(box_a, box_b)
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    min_area = max(1.0, min(area_a, area_b))

    return inter / min_area


def bbox_area(box):
    return max(0, box[2]) * max(0, box[3])


def smooth_value(prev_value, measured_value, alpha):
    if prev_value is None:
        return measured_value

    return (alpha * measured_value) + ((1.0 - alpha) * prev_value)


def smooth_box(prev_box, measured_box, alpha=BBOX_EMA_ALPHA):
    if prev_box is None:
        return tuple(float(v) for v in measured_box)

    return tuple(
        (alpha * float(measured_box[i])) + ((1.0 - alpha) * float(prev_box[i]))
        for i in range(4)
    )


def int_box(box):
    x, y, w, h = box
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def xyxy_to_xywh(x1, y1, x2, y2):
    x = int(round(x1))
    y = int(round(y1))
    w = int(round(x2 - x1))
    h = int(round(y2 - y1))

    return (x, y, w, h)


# ============================================================
# Class utilities
# ============================================================

def normalize_class_name(class_name):
    return str(class_name).strip().lower()


def get_confirmation_frames(class_name):
    if class_name in FIRE_CLASS_NAMES:
        return FIRE_CONFIRMATION_FRAMES

    if class_name in PERSON_CLASS_NAMES:
        return PERSON_CONFIRMATION_FRAMES

    return PERSON_CONFIRMATION_FRAMES


# ============================================================
# Detection
# ============================================================

def detect_objects(frame, model, valid_class_names, confidence):
    detections = []

    results = model.predict(
        frame,
        conf=confidence,
        iou=YOLO_IOU,
        imgsz=YOLO_IMAGE_SIZE,
        device=YOLO_DEVICE,
        half=YOLO_HALF,
        verbose=False,
    )

    if len(results) == 0:
        return detections

    result = results[0]
    names = result.names

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence_score = float(box.conf[0])
        class_name = normalize_class_name(names[class_id])

        if class_name not in valid_class_names:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox = xyxy_to_xywh(x1, y1, x2, y2)

        detections.append(
            {
                "bbox": bbox,
                "class_name": class_name,
                "confidence": confidence_score,
            }
        )

    return suppress_overlapping_detections(detections)


def suppress_overlapping_detections(detections, iou_threshold=DETECTION_NMS_IOU):
    if not detections:
        return detections

    kept = []
    by_class = {}

    for detection in detections:
        by_class.setdefault(detection["class_name"], []).append(detection)

    for class_name, class_detections in by_class.items():
        class_detections.sort(key=lambda item: item["confidence"], reverse=True)

        kept_class = []

        while class_detections:
            best = class_detections.pop(0)
            kept_class.append(best)

            class_detections = [
                candidate
                for candidate in class_detections
                if bbox_iou(best["bbox"], candidate["bbox"]) < iou_threshold
            ]

        if class_name in PERSON_CLASS_NAMES:
            kept_class = suppress_nested_person_boxes(kept_class)

        kept.extend(kept_class)

    return kept


def suppress_nested_person_boxes(person_detections):
    if len(person_detections) <= 1:
        return person_detections

    keep = [True] * len(person_detections)

    for i in range(len(person_detections)):
        if not keep[i]:
            continue

        box_i = person_detections[i]["bbox"]
        area_i = bbox_area(box_i)

        for j in range(len(person_detections)):
            if i == j or not keep[j]:
                continue

            box_j = person_detections[j]["bbox"]
            area_j = bbox_area(box_j)

            small_area = min(area_i, area_j)
            large_area = max(area_i, area_j)

            if large_area <= 0:
                continue

            area_ratio = small_area / large_area
            containment = bbox_iomin(box_i, box_j)

            if containment < PERSON_NESTED_IOMIN:
                continue

            if area_ratio > PERSON_NESTED_AREA_RATIO:
                continue

            # Remove the smaller nested detection.
            if area_i <= area_j:
                keep[i] = False
                break

            keep[j] = False

    return [det for idx, det in enumerate(person_detections) if keep[idx]]


# ============================================================
# Tracking
# ============================================================

def update_tracks(detections, tracks, next_track_id):
    """Update track states and return active tracks as (track_id, track)."""
    used_tracks = set()

    detections = sorted(
        detections,
        key=lambda item: item["bbox"][2] * item["bbox"][3],
        reverse=True,
    )

    for detection in detections:
        box = detection["bbox"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        center = bbox_center(box)
        _, _, w, h = box

        best_id = None
        best_score = float("inf")

        for track_id, track in tracks.items():
            if track_id in used_tracks:
                continue

            if track["class_name"] != class_name:
                continue

            prev_box = track["bbox"]
            iou = bbox_iou(tuple(box), prev_box)

            misses = track["misses"]

            adaptive_gate = max(TRACK_MAX_MATCH_DISTANCE, 1.2 * max(w, h))
            adaptive_gate *= (1.0 + 0.15 * misses)

            dist = euclidean_distance(center, track["center"])

            if dist > adaptive_gate and iou < MIN_IOU_FOR_MATCH:
                continue

            score = (1.0 - iou) + (dist / max(1.0, adaptive_gate))

            if score < best_score:
                best_score = score
                best_id = track_id

        if best_id is None:
            best_id = next_track_id
            next_track_id += 1

            smoothed_box = smooth_box(None, box)

            tracks[best_id] = {
                "center": bbox_center(smoothed_box),
                "bbox": smoothed_box,
                "class_name": class_name,
                "misses": 0,
                "hit_count": 1,
                "confirmed": get_confirmation_frames(class_name) <= 1,
                "smoothed_confidence": confidence,
            }

        else:
            smoothed_box = smooth_box(tracks[best_id]["bbox"], box)

            smoothed_confidence = smooth_value(
                tracks[best_id]["smoothed_confidence"],
                confidence,
                CONFIDENCE_EMA_ALPHA,
            )

            tracks[best_id]["center"] = bbox_center(smoothed_box)
            tracks[best_id]["bbox"] = smoothed_box
            tracks[best_id]["misses"] = 0
            tracks[best_id]["hit_count"] += 1
            tracks[best_id]["smoothed_confidence"] = smoothed_confidence

            if tracks[best_id]["hit_count"] >= get_confirmation_frames(class_name):
                tracks[best_id]["confirmed"] = True

        used_tracks.add(best_id)

    stale_ids = []

    for track_id, track in tracks.items():
        if track_id not in used_tracks:
            track["misses"] += 1

            if track["misses"] > TRACK_MAX_MISSES:
                stale_ids.append(track_id)

    for track_id in stale_ids:
        tracks.pop(track_id, None)

    active_tracks = [
        (track_id, track)
        for track_id, track in tracks.items()
        if track["misses"] <= TRACK_MAX_MISSES
    ]

    active_tracks.sort(key=lambda item: item[0])

    return active_tracks, tracks, next_track_id


# ============================================================
# Fire-person interaction
# ============================================================

def find_fire_person_interactions(active_tracks, threshold=INTERACTION_IOMIN_THRESHOLD):
    """Return list of (fire_id, person_id, overlap) that are intersecting."""
    fires = [
        (track_id, track)
        for track_id, track in active_tracks
        if track["confirmed"] and track["class_name"] in FIRE_CLASS_NAMES
    ]

    persons = [
        (track_id, track)
        for track_id, track in active_tracks
        if track["confirmed"] and track["class_name"] in PERSON_CLASS_NAMES
    ]

    interactions = []

    for fire_id, fire_track in fires:
        for person_id, person_track in persons:
            overlap = bbox_iomin(fire_track["bbox"], person_track["bbox"])

            if overlap >= threshold:
                interactions.append((fire_id, person_id, overlap))

    return interactions


# ============================================================
# Drawing
# ============================================================

def get_track_color(class_name, confirmed):
    if class_name in FIRE_CLASS_NAMES:
        return RED if confirmed else ORANGE

    if class_name in SMOKE_CLASS_NAMES:
        return YELLOW if confirmed else BLACK

    if class_name in PERSON_CLASS_NAMES:
        return GREEN if confirmed else BLUE

    return GREEN


def draw_tracks(frame, active_tracks):
    fire_count = 0
    smoke_count = 0
    person_count = 0

    for track_id, track in active_tracks:
        x, y, w, h = int_box(track["bbox"])

        if w <= 0 or h <= 0:
            continue

        class_name = track["class_name"]
        confidence = track["smoothed_confidence"]
        confirmed = track["confirmed"]

        color = get_track_color(class_name, confirmed)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        status = "CONFIRMED" if confirmed else "checking"

        label = f"ID {track_id} | {class_name} | {confidence:.2f} | {status}"

        cv2.putText(frame, label, (x, max(20, y - 10)), FONT, 0.55, color, 2,)

        if confirmed and class_name in FIRE_CLASS_NAMES:
            fire_count += 1

        if confirmed and class_name in SMOKE_CLASS_NAMES:
            smoke_count += 1

        if confirmed and class_name in PERSON_CLASS_NAMES:
            person_count += 1

    return fire_count, smoke_count, person_count


def draw_interactions(frame, active_tracks, interactions):
    track_map = dict(active_tracks)

    for fire_id, person_id, overlap in interactions:
        fire_box = int_box(track_map[fire_id]["bbox"])
        person_box = int_box(track_map[person_id]["bbox"])

        fx, fy, fw, fh = fire_box
        px, py, pw, ph = person_box

        ix1 = max(fx, px)
        iy1 = max(fy, py)
        ix2 = min(fx + fw, px + pw)
        iy2 = min(fy + fh, py + ph)

        if ix2 > ix1 and iy2 > iy1:
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), ORANGE, 3)

            cv2.putText(frame, f"INTERACT F{fire_id}-P{person_id} {overlap:.2f}",
                (ix1, max(20, iy1 - 10)), FONT, 0.55, ORANGE, 2,)


def draw_status_bar(frame, fire_count, smoke_count, person_count):
    height, width = frame.shape[:2]

    x1 = 20
    x2 = min(width - 20, 650)
    y = 30

    cv2.line(frame, (x1, y), (x2, y), RED, 30)
    cv2.line(frame, (x1, y), (x2, y), BLACK, 26)

    status_text = (
        f"Fire: {fire_count} | "
        f"Smoke: {smoke_count} | "
        f"Person: {person_count} | "
        "q quit"
    )
    cv2.putText(frame, status_text, (30, 36), FONT, 0.55, GREEN, 2,)

def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (30, 70), FONT, 0.7, GREEN, 2,)


# ============================================================
# Frame processing
# ============================================================

def process_frame(frame, detection_model, tracks, next_track_id):
    detections = detect_objects(
        frame,
        detection_model,
        VALID_CLASS_NAMES,
        YOLO_CONFIDENCE,
    )

    active_tracks, tracks, next_track_id = update_tracks(
        detections,
        tracks,
        next_track_id,
    )

    fire_count, smoke_count, person_count = draw_tracks(frame, active_tracks)

    draw_status_bar(frame, fire_count, smoke_count, person_count)

    return frame, tracks, next_track_id, fire_count, smoke_count, person_count


def draw_existing_tracks_only(frame, tracks):
    """Draw previous tracks without running YOLO detection."""
    active_tracks = [
        (track_id, track)
        for track_id, track in tracks.items()
        if track["misses"] <= TRACK_MAX_MISSES
    ]

    active_tracks.sort(key=lambda item: item[0])

    fire_count, smoke_count, person_count = draw_tracks(frame, active_tracks)

    draw_status_bar(frame, fire_count, smoke_count, person_count)

    return frame, fire_count, smoke_count, person_count


# ============================================================
# Video capture
# ============================================================

def open_video_capture(source):
    if isinstance(source, int):
        backend_candidates = []

        if os.name == "nt":
            backend_candidates.extend([cv2.CAP_DSHOW, cv2.CAP_MSMF])

        backend_candidates.append(None)

        for backend in backend_candidates:
            if backend is None:
                cap = cv2.VideoCapture(source)
            else:
                cap = cv2.VideoCapture(source, backend)

            if not cap.isOpened():
                cap.release()
                continue

            ok, _ = cap.read()

            if ok:
                return cap

            cap.release()

        raise RuntimeError(
            "Cannot open webcam source. Tried DirectShow/MSMF/default backends."
        )

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input source: {source}")

    return cap


# ============================================================
# Main loop
# ============================================================

def main():
    if not os.path.exists(DETECTION_MODEL_PATH):
        raise RuntimeError(
            f"Model not found: {DETECTION_MODEL_PATH}\n"
            "Please check DETECTION_MODEL_PATH."
        )

    print(f"Using YOLO device: {YOLO_DEVICE}")
    print(f"Using YOLO half precision: {YOLO_HALF}")
    print(f"Detection model: {DETECTION_MODEL_PATH}")

    detection_model = YOLO(DETECTION_MODEL_PATH)

    cap = open_video_capture(INPUT_SOURCE)

    # Ask camera/backend to provide lower resolution.
    # For a normal webcam, this may work.
    # For ANAFI or RTSP stream, OpenCV may ignore it,
    # so we still manually resize the frame below.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Try to reduce camera buffering latency.
    # This is backend-dependent; some cameras/streams ignore it.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    tracks = {}
    next_track_id = 1

    frame_id = 0
    prev_time = time.time()

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("Cannot read frame. Exiting.")
                break

            frame_id += 1

            # Resize incoming frame before detection and display.
            # This helps a lot for high-resolution ANAFI streams.
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Run YOLO every N frames only.
            if frame_id % PROCESS_EVERY_N_FRAMES == 0:
                frame, tracks, next_track_id, fire_count, smoke_count, person_count = process_frame(
                    frame,
                    detection_model,
                    tracks,
                    next_track_id,
                )

            else:
                # On skipped frames, draw the previous tracks only.
                # This avoids running YOLO on every frame.
                frame, fire_count, smoke_count, person_count = draw_existing_tracks_only(
                    frame,
                    tracks,
                )

            # FPS calculation.
            current_time = time.time()
            fps = 1.0 / max(1e-6, current_time - prev_time)
            prev_time = current_time

            draw_fps(frame, fps)

            cv2.imshow("Fire and Human Detector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
