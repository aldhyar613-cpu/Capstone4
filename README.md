# =============================================
#  Requirements - YOLOv11 Vehicle Detection
#  GPU   : NVIDIA GeForce RTX 5050 (Blackwell)
#  CUDA  : 12.8 (cu128)
#  Python: 3.12.10
#  OS    : Windows 11
# =============================================
#
#
#  ╔══════════════════════════════════════════╗
#  ║  BELUM PUNYA model best.pt ?             ║
#  ║  → Ikuti JALUR A (Training dulu)         ║
#  ║                                          ║
#  ║  SUDAH PUNYA model best.pt ?             ║
#  ║  → Lewati ke JALUR B (Langsung pakai)    ║
#  ╚══════════════════════════════════════════╝
#
#
# =============================================
#  JALUR A — BELUM PUNYA MODEL (perlu training)
# =============================================
#
#  Langkah A1. Buat virtual environment
#       python -m venv venv untuk .py
#       python -m venv yolo-env untuk jupyter
#
#  Langkah A2. Aktifkan virtual environment
#       .\yolo-env\Scripts\activate          (Windows)
#        .\venv\Scripts\activate
#       source yolo-env/bin/activate         (Linux/Mac)
#
#  Langkah A3. Install PyTorch versi GPU (WAJIB cu128 untuk RTX 50 series)
#       pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
#
#  Langkah A4. Install semua dependensi (termasuk Jupyter untuk training)
#       pip install -r requirements.txt
#
#  Langkah A5. Verifikasi GPU terdeteksi
#       python -c "import torch; print('GPU OK:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
#       → Harus muncul: GPU OK: True | NVIDIA GeForce RTX 5050
#       → Jika False: pastikan driver NVIDIA >= 570.xx sudah terinstall
#
#  Langkah A6. Pecah Video per Frame
#       Jika ingin mengubah video per frame masukkan video ke folder video
#       Lalu run python split_frames.py
#
#  Langkah A7. Jalankan notebook training (buka Jupyter dulu)
#       jupyter notebook
#       → Buka notebooks/train_yolov11.ipynb          (model CCTV)
#       → Buka notebooks/Train_YOLOv12_Vehicle.ipynb  (model original)
#       → Jalankan semua cell dari atas ke bawah
#       → Setelah selesai, copy hasil best.pt ke folder models/
#
#  Langkah A8. Jalankan aplikasi
#       streamlit run app.py
# 
#  Langkah A9
#        masukkan foto dan sesuaikan model sesuai kebutuhan
#         gunakan model a untuk menganalisa gambar
#         gunakan model b untuk menganalisa video sesuai video yang sudah ditrain
#
#
# =============================================
#  JALUR B — SUDAH PUNYA MODEL (langsung pakai)
# =============================================
#
#  Langkah B1. Pastikan file model ada di folder models/
#       models/best.pt          ← model original (bus, car, van)
#       models/best_video.pt    ← model CCTV (car, motorcycle, truck)
#
#  Langkah B2. Buat virtual environment
#       python -m venv yolo-env untuk jupyter
#       python -m venv venv untuk .py
#
#  Langkah B3. Aktifkan virtual environment
#       .\yolo-env\Scripts\activate          (Windows)
#       .\venv\Scripts\activate khusus run.py
#       source yolo-env/bin/activate         (Linux/Mac)
#
#  Langkah B4. Install PyTorch
#
#       [OPSI 1 — Pakai GPU, komputer ada NVIDIA RTX 50 series]
#  pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
#
#       [OPSI 2 — Tanpa GPU / komputer lain / server / deploy]
#       pip install torch torchvision torchaudio
#
#  Langkah B5. Install dependensi aplikasi
#       pip install -r requirements.txt
#
#  Langkah B6. Jalankan aplikasi
#       streamlit run app.py
#       → Buka browser: http://localhost:8501
#
#  Langkah
#        masukkan foto dan sesuaikan model sesuai kebutuhan
#         gunakan model a untuk menganalisa gambar
#         gunakan model b untuk menganalisa video sesuai video yang sudah ditrain
# =============================================
#  CATATAN PENTING
# =============================================
#  - RTX 50 series (Blackwell) WAJIB pakai cu128, jangan cu118/cu121
#  - Pastikan NVIDIA Driver >= 570.xx sudah terinstall sebelum install PyTorch GPU
#  - PyTorch tidak dimasukkan ke file ini karena versinya berbeda
#    tergantung GPU yang dipakai (lihat langkah A3 atau B4 di atas)
#  - Jupyter hanya dibutuhkan di Jalur A (training), tidak wajib di Jalur B
# =============================================


# ----- Core YOLO & Computer Vision -----
ultralytics==8.4.54
supervision==0.28.0

# ----- Web App -----
streamlit>=1.32.0

# ----- Visualisasi Analytics -----
plotly>=5.0.0
pandas>=2.0.0

# ----- Jupyter (hanya untuk Jalur A / training, bisa skip di Jalur B) -----
jupyter==1.1.1
ipykernel==6.29.5