"""
Realtime webcam detection and counting for fire, smoke, and humans.

Model expected: Ultralytics YOLO checkpoint trained with classes:
    0: fire
    1: human
    2: smoke

Run:
    pip install ultralytics opencv-python numpy
    python webcam_fire_smoke_human_counter.py --model firesmokehuman.pt --camera 0

Example with class-specific confidence thresholds:
    python webcam_fire_smoke_human_counter.py --model firesmokehuman.pt --camera 0 \
        --conf-human 0.40 --conf-fire 0.30 --conf-smoke 0.25

Keys:
    q / ESC : quit
    s       : save current annotated frame
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Ultralytics is not installed. Install it first with:\n"
        "    pip install ultralytics opencv-python numpy\n"
    ) from exc


BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    box: BBox
    index: int


CLASS_ALIASES = {
    "fire": {"fire", "flame", "flames"},
    "human": {"human", "person", "people", "man", "woman"},
    "smoke": {"smoke", "smog"},
}

COLORS = {
    "fire": (0, 0, 255),      # red in BGR
    "human": (0, 255, 0),    # green in BGR
    "smoke": (160, 160, 160),# gray in BGR
    "human_fire": (0, 255, 255),  # yellow in BGR
    "text_bg": (20, 20, 20),
    "white": (255, 255, 255),
}


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def canonical_class(name: str) -> str | None:
    n = normalize_name(name)
    for canonical, aliases in CLASS_ALIASES.items():
        if n in aliases:
            return canonical
    return None


def area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def center_in_box(inner: BBox, outer: BBox) -> bool:
    x1, y1, x2, y2 = inner
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    ox1, oy1, ox2, oy2 = outer
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def human_in_fire(human_box: BBox, fire_box: BBox, min_overlap_human: float) -> bool:
    """
    Determine whether a detected human is inside / affected by a fire region.

    IoU alone is often too strict because a fire box can be large or partial.
    Therefore we use two robust criteria:
      1. the human box center lies inside the fire box, or
      2. intersection_area / human_area >= min_overlap_human.
    """
    h_area = area(human_box)
    if h_area <= 0:
        return False
    overlap_ratio_over_human = intersection_area(human_box, fire_box) / h_area
    return center_in_box(human_box, fire_box) or overlap_ratio_over_human >= min_overlap_human


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y_top = max(0, y - th - baseline - 4)
    cv2.rectangle(frame, (x, y_top), (x + tw + 8, y_top + th + baseline + 6), color, -1)
    cv2.putText(frame, text, (x + 4, y_top + th + 2), font, scale, COLORS["white"], thickness, cv2.LINE_AA)


def draw_box(frame: np.ndarray, box: BBox, label: str, color: Tuple[int, int, int], thickness: int = 2) -> None:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    draw_label(frame, label, x1, y1, color)


def parse_detections(result, model_names: Dict[int, str]) -> List[Detection]:
    detections: List[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    for idx, (box, conf, cls_id) in enumerate(zip(xyxy, confs, cls_ids)):
        raw_name = model_names.get(int(cls_id), str(cls_id))
        canonical = canonical_class(raw_name)
        if canonical is None:
            continue
        detections.append(
            Detection(
                cls_id=int(cls_id),
                cls_name=canonical,
                conf=float(conf),
                box=tuple(float(v) for v in box),
                index=idx,
            )
        )
    return detections


def filter_by_class_conf(detections: List[Detection], args: argparse.Namespace) -> List[Detection]:
    """Apply independent confidence thresholds after YOLO inference.
    To avoid losing low-confidence detections of one class just because another class has a higher threshold, we set a low global confidence for YOLO and then filter by class here.
    """
    thresholds = {
        "human": float(args.conf_human),
        "fire": float(args.conf_fire),
        "smoke": float(args.conf_smoke),
    }
    return [d for d in detections if d.conf >= thresholds.get(d.cls_name, 1.0)]


def count_fire_human_overlap(
    humans: List[Detection],
    fires: List[Detection],
    min_overlap_human: float,
) -> Tuple[Dict[int, List[int]], List[int]]:
    """
    Returns:
        fire_to_humans: {fire_detection_index: [human_detection_index, ...]}
        humans_in_any_fire: unique human detection indices affected by at least one fire box
    """
    fire_to_humans: Dict[int, List[int]] = {f.index: [] for f in fires}
    humans_in_any_fire = set()

    for fire in fires:
        for human in humans:
            if human_in_fire(human.box, fire.box, min_overlap_human):
                fire_to_humans[fire.index].append(human.index)
                humans_in_any_fire.add(human.index)

    return fire_to_humans, sorted(humans_in_any_fire)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    total_humans: int,
    total_fires: int,
    total_smoke: int,
    humans_in_fire: int,
) -> None:
    lines = [
        f"FPS: {fps:.1f}",
        f"Humans: {total_humans}",
        f"Fire boxes: {total_fires}",
        f"Smoke boxes: {total_smoke}",
        f"Humans in fire: {humans_in_fire}",
    ]
    x, y = 12, 24
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS["white"], 2, cv2.LINE_AA)
        y += 26


def save_json_log(log_path: Path, record: dict) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = YOLO(str(model_path))
    print("Loaded model classes:", model.names)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    prev_time = time.perf_counter()
    fps = 0.0
    frame_id = 0
    log_path = Path(args.log_jsonl) if args.log_jsonl else None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        frame_id += 1
        # Use a low global pre-filter confidence for YOLO, then apply
        # independent class-specific thresholds after parsing detections.
        # This prevents, for example, smoke at 0.25 from being discarded
        # just because human is set to 0.40.
        inference_conf = min(args.conf, args.conf_human, args.conf_fire, args.conf_smoke)

        result = model.predict(
            source=frame,
            conf=inference_conf,
            iou=args.nms_iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
            agnostic_nms=False,  # important: keep overlapping fire/human boxes from different classes
            max_det=args.max_det,
        )[0]

        detections = parse_detections(result, model.names)
        detections = filter_by_class_conf(detections, args)

        humans = [d for d in detections if d.cls_name == "human"]
        fires = [d for d in detections if d.cls_name == "fire"]
        smokes = [d for d in detections if d.cls_name == "smoke"]

        fire_to_humans, humans_in_any_fire = count_fire_human_overlap(
            humans=humans,
            fires=fires,
            min_overlap_human=args.min_overlap_human,
        )
        humans_in_fire_set = set(humans_in_any_fire)

        # Draw fire and smoke first, then humans so human boxes remain readable.
        for fire_idx, fire in enumerate(fires, start=1):
            n_humans = len(fire_to_humans.get(fire.index, []))
            label = f"fire F{fire_idx} {fire.conf:.2f} | humans: {n_humans}"
            draw_box(frame, fire.box, label, COLORS["fire"], thickness=2)

        for smoke_idx, smoke in enumerate(smokes, start=1):
            label = f"smoke S{smoke_idx} {smoke.conf:.2f}"
            draw_box(frame, smoke.box, label, COLORS["smoke"], thickness=2)

        for human_idx, human in enumerate(humans, start=1):
            in_fire = human.index in humans_in_fire_set
            color = COLORS["human_fire"] if in_fire else COLORS["human"]
            suffix = " IN_FIRE" if in_fire else ""
            label = f"human H{human_idx} {human.conf:.2f}{suffix}"
            draw_box(frame, human.box, label, color, thickness=3 if in_fire else 2)

        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

        draw_hud(
            frame,
            fps=fps,
            total_humans=len(humans),
            total_fires=len(fires),
            total_smoke=len(smokes),
            humans_in_fire=len(humans_in_any_fire),
        )

        if log_path is not None:
            save_json_log(
                log_path,
                {
                    "frame_id": frame_id,
                    "time_unix": time.time(),
                    "num_humans": len(humans),
                    "num_fires": len(fires),
                    "num_smoke": len(smokes),
                    "num_humans_in_fire": len(humans_in_any_fire),
                    "fire_to_humans": fire_to_humans,
                },
            )

        cv2.imshow("Fire-Smoke-Human Webcam Counter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            out = Path(f"annotated_frame_{frame_id:06d}.jpg")
            cv2.imwrite(str(out), frame)
            print(f"Saved {out}")

    cap.release()
    cv2.destroyAllWindows()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime webcam counter for fire, smoke, and humans.")
    parser.add_argument("--model", type=str, default="firesmokehuman.pt", help="Path to YOLO .pt model.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index. Usually 0 for laptop webcam.")
    parser.add_argument("--width", type=int, default=1280, help="Requested webcam width.")
    parser.add_argument("--height", type=int, default=720, help="Requested webcam height.")
    parser.add_argument("--imgsz", type=int, default=480, help="YOLO inference image size.")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help=(
            "Low global YOLO pre-filter confidence. Final filtering is done by "
            "--conf-human, --conf-fire, and --conf-smoke. Usually leave this low."
        ),
    )
    parser.add_argument("--conf-human", type=float, default=0.35, help="Final confidence threshold for human detections.")
    parser.add_argument("--conf-fire", type=float, default=0.30, help="Final confidence threshold for fire detections.")
    parser.add_argument("--conf-smoke", type=float, default=0.25, help="Final confidence threshold for smoke detections.")
    parser.add_argument("--nms-iou", type=float, default=0.50, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per frame.")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, 0, 0,1, etc. Leave empty for auto.")
    parser.add_argument(
        "--min-overlap-human",
        type=float,
        default=0.10,
        help="Minimum intersection/human-area ratio to classify a human as inside/affected by fire.",
    )
    parser.add_argument(
        "--log-jsonl",
        type=str,
        default="",
        help="Optional JSONL output path for per-frame counts.",
    )
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
