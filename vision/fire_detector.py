"""Real-time fire and smoke detection using YOLO.

Controls:
- q: quit

Notes:
- Use a custom YOLO model trained for fire/smoke detection.
- Set INPUT_SOURCE = 0 for webcam.
- Set INPUT_SOURCE = "fire_image.jpg" for image.
- Set INPUT_SOURCE = "fire_video.mp4" for video.
"""

import cv2
import math
import os
from ultralytics import YOLO


# Input source.
# 0 for laptop webcam.
INPUT_SOURCE = 0

# Custom YOLO model trained for fire/smoke.
# Do not use normal COCO weights unless your model has fire/smoke classes.
# Trained model from local run.
FIRE_MODEL_PATH = "runs/detect/train/weights/best.pt"

# YOLO inference parameters.
YOLO_CONFIDENCE = 0.45
YOLO_IMAGE_SIZE = 640

# Fire/smoke class names expected from the custom model.
FIRE_CLASS_NAMES = ("fire", "flame")
SMOKE_CLASS_NAMES = ("smoke",)
VALID_CLASS_NAMES = FIRE_CLASS_NAMES + SMOKE_CLASS_NAMES

# Set True when using your trained fire/smoke model.
ENABLE_CLASS_FILTER = True

# Tracking parameters.
TRACK_MAX_MISSES = 15
TRACK_MAX_MATCH_DISTANCE = 120
BBOX_EMA_ALPHA = 0.35
CONFIDENCE_EMA_ALPHA = 0.30
MIN_IOU_FOR_MATCH = 0.05

# Temporal confirmation.
# Fire/smoke must be detected for several frames before being confirmed.
CONFIRMATION_FRAMES = 5

GREEN = (0, 255, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
YELLOW = (0, 255, 255)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_COMPLEX


def is_image_file(source):
    if not isinstance(source, str):
        return False

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return source.lower().endswith(image_extensions)


def clamp(value, low, high):
    return max(low, min(value, high))


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


def normalize_class_name(class_name):
    return str(class_name).strip().lower()


def detect_fire_objects(frame, model):
    detections = []

    results = model.predict(
        frame,
        conf=YOLO_CONFIDENCE,
        imgsz=YOLO_IMAGE_SIZE,
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
        confidence = float(box.conf[0])
        class_name = normalize_class_name(names[class_id])

        if ENABLE_CLASS_FILTER and class_name not in VALID_CLASS_NAMES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox = xyxy_to_xywh(x1, y1, x2, y2)

        detections.append(
            {
                "bbox": bbox,
                "class_name": class_name,
                "confidence": confidence,
            }
        )

    return detections


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
                "confirmed": False,
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

            if tracks[best_id]["hit_count"] >= CONFIRMATION_FRAMES:
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


def draw_tracks(frame, active_tracks):
    fire_confirmed = False
    smoke_confirmed = False

    for track_id, track in active_tracks:
        x, y, w, h = int_box(track["bbox"])

        if w <= 0 or h <= 0:
            continue

        class_name = track["class_name"]
        confidence = track["smoothed_confidence"]
        confirmed = track["confirmed"]

        if class_name in FIRE_CLASS_NAMES:
            color = RED if confirmed else ORANGE
        else:
            color = YELLOW if confirmed else GREEN

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        if confirmed:
            status = "CONFIRMED"
        else:
            status = "checking"

        label = f"ID {track_id} | {class_name} | {confidence:.2f} | {status}"

        cv2.putText(
            frame,
            label,
            (x, max(20, y - 10)),
            FONT,
            0.55,
            color,
            2,
        )

        if confirmed and class_name in FIRE_CLASS_NAMES:
            fire_confirmed = True

        if confirmed and class_name in SMOKE_CLASS_NAMES:
            smoke_confirmed = True

    return fire_confirmed, smoke_confirmed


def draw_status_bar(frame, fire_confirmed, smoke_confirmed):
    cv2.line(frame, (20, 30), (620, 30), RED, 30)
    cv2.line(frame, (20, 30), (620, 30), BLACK, 26)

    if fire_confirmed:
        status_text = "FIRE CONFIRMED | q quit"
        status_color = RED
    elif smoke_confirmed:
        status_text = "SMOKE CONFIRMED | q quit"
        status_color = YELLOW
    else:
        status_text = "Monitoring fire/smoke | q quit"
        status_color = GREEN

    cv2.putText(
        frame,
        status_text,
        (30, 36),
        FONT,
        0.55,
        status_color,
        2,
    )


def process_frame(frame, model, tracks, next_track_id):
    detections = detect_fire_objects(frame, model)
    active_tracks, tracks, next_track_id = update_tracks(
        detections,
        tracks,
        next_track_id,
    )

    fire_confirmed, smoke_confirmed = draw_tracks(frame, active_tracks)
    draw_status_bar(frame, fire_confirmed, smoke_confirmed)

    return frame, tracks, next_track_id, fire_confirmed, smoke_confirmed


def run_image_source(model):
    frame = cv2.imread(INPUT_SOURCE)
    if frame is None:
        raise RuntimeError(f"Cannot open image: {INPUT_SOURCE}")

    tracks = {}
    next_track_id = 1

    frame, tracks, next_track_id, fire_confirmed, smoke_confirmed = process_frame(
        frame,
        model,
        tracks,
        next_track_id,
    )

    print(f"Fire confirmed: {fire_confirmed}")
    print(f"Smoke confirmed: {smoke_confirmed}")

    while True:
        cv2.imshow("Fire and Smoke Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


def run_video_source(model):
    cap = cv2.VideoCapture(INPUT_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input source: {INPUT_SOURCE}")

    tracks = {}
    next_track_id = 1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame, tracks, next_track_id, fire_confirmed, smoke_confirmed = process_frame(
                frame,
                model,
                tracks,
                next_track_id,
            )

            if fire_confirmed:
                print("Fire confirmed.")
            elif smoke_confirmed:
                print("Smoke confirmed.")

            cv2.imshow("Fire and Smoke Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()


def main():
    if not os.path.exists(FIRE_MODEL_PATH):
        raise RuntimeError(
            f"Model file not found: {FIRE_MODEL_PATH}\n"
            "Please set FIRE_MODEL_PATH to your custom fire/smoke YOLO model."
        )

    model = YOLO(FIRE_MODEL_PATH)

    try:
        if is_image_file(INPUT_SOURCE):
            run_image_source(model)
        else:
            run_video_source(model)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()