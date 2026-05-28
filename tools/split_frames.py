"""
tools/split_frames.py — Pecah Video CCTV Per Frame
===================================================
Skrip utilitas untuk memecah video menjadi frame-frame gambar.
Hasil frame disimpan otomatis ke folder tools/frames/.

Cara pakai:
  1. Taruh video di folder: tools/video/
  2. Jalankan dari root project:
         python tools/split_frames.py
  3. Hasil frame tersimpan di: tools/frames/

Struktur folder:
  capstone_vehicle_detection/
  ├── tools/
  │   ├── split_frames.py      ← file ini
  │   ├── video/               ← taruh video input di sini
  │   │   └── cctv_kendaraan.mp4
  │   └── frames/              ← dibuat otomatis, hasil frame
  └── ...
"""

import cv2
import os


# ─────────────────────────────────────────────
# KONFIGURASI — Ubah sesuai kebutuhan
# ─────────────────────────────────────────────

# Direktori tools/ (tempat file ini berada)
TOOLS_DIR     = os.path.dirname(os.path.abspath(__file__))

# Folder input video dan output frames
VIDEO_FOLDER  = os.path.join(TOOLS_DIR, "video")
OUTPUT_FOLDER = os.path.join(TOOLS_DIR, "frames")

# Ambil frame setiap N frame sekali (10 = setiap 10 frame)
INTERVAL      = 10


# ─────────────────────────────────────────────
# FUNGSI UTAMA
# ─────────────────────────────────────────────

def split_video_to_frames(
    video_path   : str,
    output_folder: str,
    interval     : int
) -> None:
    """
    Memecah video menjadi frame-frame gambar dan menyimpannya ke folder.

    Parameter:
        video_path    : path lengkap ke file video input
        output_folder : path folder untuk menyimpan hasil frame
        interval      : ambil 1 frame setiap N frame (contoh: 10 = setiap 10 frame)
    """

    # ── Cek file video ada ───────────────────
    if not os.path.exists(video_path):
        print(f"[ERROR] File video tidak ditemukan: {video_path}")
        print(f"        Pastikan video ada di folder tools/video/")
        return

    # ── Buat folder output kalau belum ada ───
    os.makedirs(output_folder, exist_ok=True)
    print(f"[INFO] Folder output: {output_folder}/")

    # ── Buka video ───────────────────────────
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Tidak bisa membuka video: {video_path}")
        return

    # ── Info video ───────────────────────────
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps           = cap.get(cv2.CAP_PROP_FPS)
    durasi_detik  = total_frames / fps if fps > 0 else 0
    width         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 50)
    print("  INFO VIDEO")
    print("=" * 50)
    print(f"  File       : {video_path}")
    print(f"  Resolusi   : {width} x {height} px")
    print(f"  FPS        : {fps:.1f}")
    print(f"  Total frame: {total_frames}")
    print(f"  Durasi     : {durasi_detik:.1f} detik ({durasi_detik/60:.1f} menit)")
    print(f"  Interval   : setiap {interval} frame")
    print(f"  Estimasi hasil: ~{total_frames // interval} frame")
    print("=" * 50)

    # ── Proses pecah frame ───────────────────
    frame_count = 0   # counter frame video
    saved_count = 0   # counter frame yang disimpan

    print("\n[INFO] Memulai proses pecah frame...")

    while True:
        ret, frame = cap.read()

        # Berhenti kalau video habis
        if not ret:
            break

        # Simpan hanya setiap N frame
        if frame_count % interval == 0:
            # Nama file: frame_00000.jpg, frame_00010.jpg, dst
            filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

            # Tampilkan progress setiap 20 frame tersimpan
            if saved_count % 20 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  [Progress] {progress:.1f}% — {saved_count} frame tersimpan...")

        frame_count += 1

    # ── Selesai ──────────────────────────────
    cap.release()

    print("\n" + "=" * 50)
    print("  SELESAI!")
    print("=" * 50)
    print(f"  Total frame diproses : {frame_count}")
    print(f"  Total frame disimpan : {saved_count}")
    print(f"  Lokasi hasil         : {output_folder}/")
    print("=" * 50)
    print(f"\n✅ {saved_count} frame siap di-upload ke Roboflow untuk labeling!")


def find_video_file(video_folder: str) -> str | None:
    """
    Cari file video pertama yang ditemukan di dalam folder.

    Parameter:
        video_folder : path folder tempat mencari video

    Return:
        path lengkap ke file video, atau None jika tidak ditemukan
    """
    if not os.path.exists(video_folder):
        return None

    supported_ext = (".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV")
    for filename in os.listdir(video_folder):
        if filename.endswith(supported_ext):
            return os.path.join(video_folder, filename)
    return None


# ─────────────────────────────────────────────
# JALANKAN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Cari file video di folder tools/video/
    video_path = find_video_file(VIDEO_FOLDER)

    if video_path is None:
        print(f"[ERROR] Tidak ada file video di folder: {VIDEO_FOLDER}")
        print(f"        Format yang didukung: .mp4, .avi, .mov")
        print(f"        Taruh video kamu di folder tools/video/ terlebih dahulu.")
    else:
        print(f"[INFO] Video ditemukan: {video_path}")
        split_video_to_frames(
            video_path    = video_path,
            output_folder = OUTPUT_FOLDER,
            interval      = INTERVAL
        )
