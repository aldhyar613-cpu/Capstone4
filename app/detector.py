"""
app/detector.py — Logika Deteksi Kendaraan menggunakan YOLOv11 (Dynamic Rendering)
==============================================================================
Berisi semua fungsi inti untuk:
  - Load model YOLO dari file .pt
  - Deteksi kendaraan pada gambar (PIL Image) dengan bounding box proporsional
  - Deteksi + counting kendaraan pada video dengan skala teks dinamis
  - Gambar bounding box di atas gambar
"""

from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import time

from app.config import get_config


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

def load_model(model_path: str) -> YOLO:
    """
    Load model YOLO dari file .pt
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"File '{model_path}' tidak ditemukan.\n"
            f"Pastikan file .pt ada di folder models/ yang sama."
        )
    print(f"[INFO] Loading model dari: {model_path}")
    model = YOLO(model_path)
    print(f"[INFO] Model berhasil dimuat!")
    print(f"[INFO] Class yang dikenali: {list(model.names.values())}")
    return model


# ─────────────────────────────────────────────
# DETEKSI GAMBAR
# ─────────────────────────────────────────────

def detect_vehicles(
    model        : YOLO,
    image        : Image.Image,
    confidence   : float = 0.5,
    iou_threshold: float = 0.45,
    model_type   : str   = "original"
) -> dict:
    """
    Mendeteksi kendaraan dalam sebuah gambar (PIL Image).
    """
    config       = get_config(model_type)
    class_list   = config["classes"]
    class_colors = config["colors"]

    # ── Inferensi ────────────────────────────
    results     = model(image, conf=confidence, iou=iou_threshold, verbose=False)[0]
    class_names = results.names
    boxes       = results.boxes

    # ── Hitung per kelas ──────────────────────
    counts     = {cls: 0 for cls in class_list}
    detections = []

    for box in boxes:
        cls_id   = int(box.cls)
        cls_name = class_names[cls_id]
        conf_val = float(box.conf)
        xyxy     = box.xyxy[0].tolist()

        if cls_name in counts:
            counts[cls_name] += 1

        detections.append({
            "class"     : cls_name,
            "confidence": round(conf_val, 3),
            "bbox"      : [round(v, 1) for v in xyxy],
        })

    annotated_image = draw_boxes(image, detections, class_colors)
    total           = sum(counts.values())

    return {
        "annotated_image": annotated_image,
        "counts"         : counts,
        "total"          : total,
        "detections"     : detections,
        "raw_results"    : results,
    }


# ─────────────────────────────────────────────
# DETEKSI VIDEO
# ─────────────────────────────────────────────

def detect_vehicles_video(
    model             : YOLO,
    video_path        : str,
    output_path       : str,
    confidence        : float = 0.5,
    iou_threshold     : float = 0.45,
    model_type        : str   = "cctv",
    line_position     : float = 0.5,   # Tidak dipakai lagi
    progress_callback         = None
) -> dict:
    """
    Mendeteksi kendaraan dalam video dengan skala rendering yang dinamis.
    """
    config       = get_config(model_type)
    class_list   = config["classes"]
    class_colors = config["colors"]

    # ── Buka video ───────────────────────────
    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Setup video writer ───────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))

    # ── Kalkulasi Ukuran Grafis Dinamis Berdasarkan Resolusi Video ──
    base_size = max(width, height)
    thickness = max(2, int(base_size / 700))
    font_scale = max(0.4, base_size / 1800)
    font_thickness = max(1, int(base_size / 1200))
    
    # Skala untuk papan skor teks di pojok kiri atas video
    overlay_scale = max(0.5, base_size / 1600)
    overlay_thickness = max(1, int(base_size / 1100))
    line_step = int(42 * overlay_scale)

    # ── Counter kendaraan ────────────────────
    total_counts = {cls: 0 for cls in class_list}
    counted_ids  = set()
    frame_count  = 0
    fps_list     = []

    print(f"[INFO] Memproses video: {total_frames} frame dengan skala dinamis...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t_start      = time.time()

        # ── Inferensi + tracking ──────────────
        results = model.track(
            frame,
            conf    = confidence,
            iou     = iou_threshold,
            persist = True,
            verbose = False
        )[0]

        class_names = results.names
        boxes       = results.boxes

        # ── Proses tiap bounding box ──────────
        for box in boxes:
            cls_id   = int(box.cls)
            cls_name = class_names[cls_id]
            conf_val = float(box.conf)
            xyxy     = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]

            track_id = int(box.id[0]) if box.id is not None else None

            # ── Vehicle Counting ──
            if track_id is not None and cls_name in total_counts:
                key = (track_id, cls_name)
                if key not in counted_ids:
                    total_counts[cls_name] += 1
                    counted_ids.add(key)

            # ── Gambar bounding box dengan ketebalan dinamis ──
            color = class_colors.get(cls_name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Label teks
            label = f"{cls_name} {conf_val:.2f}"
            if track_id is not None:
                label = f"#{track_id} {label}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            y1_label = max(y1, th + int(10 * font_scale))

            # Gambar background label & teks dinamis
            cv2.rectangle(frame, (x1, y1_label - th - int(8 * font_scale)), (x1 + tw + int(4 * font_scale), y1_label), color, -1)
            cv2.putText(frame, label, (x1 + int(2 * font_scale), y1_label - int(4 * font_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        # ── Hitung FPS ────────────────────────
        t_end   = time.time()
        fps_cur = 1.0 / (t_end - t_start + 1e-9)
        fps_list.append(fps_cur)

        # ── Papan Skor Teks di Pojok Kiri Atas (Dinamis) ──
        overlay_y = line_step
        cv2.putText(frame, f"FPS: {fps_cur:.1f}", (int(15 * overlay_scale), overlay_y),
                    cv2.FONT_HERSHEY_SIMPLEX, overlay_scale, (255, 255, 255), overlay_thickness)
        overlay_y += line_step

        for cls in class_list:
            color = class_colors.get(cls, (255, 255, 255))
            label = f"{cls}: {total_counts[cls]}"
            cv2.putText(frame, label, (int(15 * overlay_scale), overlay_y),
                        cv2.FONT_HERSHEY_SIMPLEX, overlay_scale, color, overlay_thickness)
            overlay_y += line_step

        total_now = sum(total_counts.values())
        cv2.putText(frame, f"TOTAL: {total_now}", (int(15 * overlay_scale), overlay_y),
                    cv2.FONT_HERSHEY_SIMPLEX, overlay_scale * 1.1, (255, 255, 0), overlay_thickness + 1)

        # ── Tulis frame ke output video ───────
        out.write(frame)

        if progress_callback:
            progress_callback(frame_count / total_frames)

    cap.release()
    out.release()

    fps_avg = sum(fps_list) / len(fps_list) if fps_list else 0

    return {
        "total_counts": total_counts,
        "total"       : sum(total_counts.values()),
        "fps_avg"     : round(fps_avg, 1),
        "output_path" : output_path,
        "frame_count" : frame_count,
    }


# ─────────────────────────────────────────────
# GAMBAR BOUNDING BOX (untuk gambar statis)
# ─────────────────────────────────────────────

def draw_boxes(
    image       : Image.Image,
    detections  : list,
    class_colors: dict
) -> Image.Image:
    """
    Gambar bounding box dan label di atas gambar PIL secara dinamis.
    """
    img_array = np.array(image)
    img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Hitung skala dinamis berdasarkan sisi terpanjang gambar
    height, width, _ = img_bgr.shape
    base_size = max(width, height)
    
    thickness      = max(2, int(base_size / 700))
    font_scale     = max(0.4, base_size / 1800)
    font_thickness = max(1, int(base_size / 1200))

    for det in detections:
        cls_name        = det["class"]
        conf            = det["confidence"]
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color           = class_colors.get(cls_name, (255, 255, 255))

        # Gambar kotak dinamis
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)

        # Label background + teks dinamis
        label       = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        y1_label    = max(y1, th + int(10 * font_scale))

        cv2.rectangle(img_bgr, (x1, y1_label - th - int(8 * font_scale)), (x1 + tw + int(4 * font_scale), y1_label), color, -1)
        cv2.putText(img_bgr, label, (x1 + int(2 * font_scale), y1_label - int(4 * font_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)