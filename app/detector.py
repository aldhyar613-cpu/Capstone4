"""
app/detector.py — Logika Deteksi Kendaraan menggunakan YOLOv11
==============================================================
Berisi semua fungsi inti untuk:
  - Load model YOLO dari file .pt
  - Deteksi kendaraan pada gambar (PIL Image)
  - Deteksi + counting kendaraan pada video
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

    Parameter:
        model_path : path lengkap ke file .pt

    Return:
        objek YOLO yang sudah dimuat

    Raises:
        FileNotFoundError jika file .pt tidak ditemukan
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

    Parameter:
        model         : objek YOLO yang sudah dimuat
        image         : gambar input sebagai PIL Image (RGB)
        confidence    : confidence threshold (0.0 - 1.0)
        iou_threshold : IoU threshold untuk NMS (0.0 - 1.0)
        model_type    : "original" atau "cctv"

    Return:
        dict dengan key:
            annotated_image : PIL Image hasil deteksi dengan bounding box
            counts          : dict {nama_class: jumlah_deteksi}
            total           : int total semua deteksi
            detections      : list of dict per objek terdeteksi
            raw_results     : objek hasil ultralytics mentah
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
    line_position     : float = 0.5,   # tidak dipakai lagi, dibiarkan agar signature tidak berubah
    progress_callback         = None
) -> dict:
    """
    Mendeteksi kendaraan dalam video dengan vehicle counting.

    Counting sederhana: setiap (track_id, class) dihitung tepat 1 kali
    sejak pertama kali terdeteksi. Tidak ada garis counting.

    Parameter:
        model             : objek YOLO yang sudah dimuat
        video_path        : path ke file video input
        output_path       : path untuk menyimpan video hasil deteksi
        confidence        : confidence threshold (0.0 - 1.0)
        iou_threshold     : IoU threshold untuk NMS (0.0 - 1.0)
        model_type        : "original" atau "cctv"
        line_position     : tidak dipakai (dibiarkan agar tidak breaking change)
        progress_callback : fungsi callback(float) untuk update progress bar,
                            dipanggil dengan nilai 0.0 - 1.0

    Return:
        dict dengan key:
            total_counts : dict {nama_class: total_dihitung}
            total        : int total semua kendaraan
            fps_avg      : float rata-rata FPS pemrosesan
            output_path  : string path video output
            frame_count  : int jumlah frame yang diproses
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

    # ── Counter kendaraan ────────────────────
    total_counts = {cls: 0 for cls in class_list}
    # Set berisi tuple (track_id, cls_name) yang sudah dihitung
    counted_ids  = set()
    frame_count  = 0
    fps_list     = []

    print(f"[INFO] Memproses video: {total_frames} frame...")

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

            # Track ID (bisa None kalau tracking gagal)
            track_id = int(box.id[0]) if box.id is not None else None

            # ── Vehicle Counting (sederhana, tanpa garis) ──
            # Hitung sekali per kombinasi (track_id, class)
            if track_id is not None and cls_name in total_counts:
                key = (track_id, cls_name)
                if key not in counted_ids:
                    total_counts[cls_name] += 1
                    counted_ids.add(key)

            # ── Gambar bounding box ───────────
            color = class_colors.get(cls_name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label dengan track ID (jika ada)
            label = f"{cls_name} {conf_val:.2f}"
            if track_id is not None:
                label = f"#{track_id} {label}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ── Hitung FPS ────────────────────────
        t_end   = time.time()
        fps_cur = 1.0 / (t_end - t_start + 1e-9)
        fps_list.append(fps_cur)

        # ── Overlay counter di pojok kiri atas ─
        overlay_y = 30
        cv2.putText(frame, f"FPS: {fps_cur:.1f}", (10, overlay_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        overlay_y += 30

        for cls in class_list:
            color = class_colors.get(cls, (255, 255, 255))
            label = f"{cls}: {total_counts[cls]}"
            cv2.putText(frame, label, (10, overlay_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            overlay_y += 28

        # Total kendaraan sampai frame ini
        total_now = sum(total_counts.values())
        cv2.putText(frame, f"TOTAL: {total_now}", (10, overlay_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # ── Tulis frame ke output video ───────
        out.write(frame)

        # ── Update progress callback ──────────
        if progress_callback:
            progress_callback(frame_count / total_frames)

    # ── Selesai, release resource ─────────────
    cap.release()
    out.release()

    fps_avg = sum(fps_list) / len(fps_list) if fps_list else 0

    print(f"[INFO] Video selesai diproses!")
    print(f"[INFO] Output: {output_path}")
    print(f"[RESULT] Total: {total_counts}")

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
    Gambar bounding box dan label di atas gambar PIL.

    Parameter:
        image        : gambar input sebagai PIL Image (RGB)
        detections   : list of dict hasil detect_vehicles()
                       setiap dict berisi: class, confidence, bbox
        class_colors : dict {nama_class: tuple_BGR}

    Return:
        PIL Image (RGB) dengan bounding box dan label tergambar
    """
    img_array = np.array(image)
    img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    for det in detections:
        cls_name        = det["class"]
        conf            = det["confidence"]
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color           = class_colors.get(cls_name, (255, 255, 255))

        # Gambar kotak
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

        # Label background + teks
        label       = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img_bgr, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)