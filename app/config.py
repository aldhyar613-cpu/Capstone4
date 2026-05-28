"""
app/config.py — Konfigurasi Model & Kelas Kendaraan
====================================================
Semua konfigurasi model (path, class, warna, emoji) terpusat di sini.
Untuk menambah/mengubah model, cukup edit file ini saja.
"""

import os

# ─────────────────────────────────────────────
# BASE PATH
# ─────────────────────────────────────────────

# Direktori root project (satu level di atas folder app/)
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ─────────────────────────────────────────────
# KONFIGURASI PER MODEL
# ─────────────────────────────────────────────

MODEL_CONFIGS = {
    "original": {
        "path"   : os.path.join(MODELS_DIR, "best.pt"),
        "classes": ["bus", "car", "van"],
        "colors" : {
            "bus": (0, 165, 255),    # orange
            "car": (0, 255, 0),      # hijau
            "van": (255, 0, 128),    # ungu
        },
        "emojis" : {"bus": "🚌", "car": "🚗", "van": "🚐"},
    },
    "cctv": {
        "path"   : os.path.join(MODELS_DIR, "best_video.pt"),
        "classes": ["car", "motorcycle", "truck"],
        "colors" : {
            "car"       : (0, 255, 0),      # hijau
            "motorcycle": (255, 165, 0),    # biru
            "truck"     : (0, 0, 255),      # merah
        },
        "emojis" : {"car": "🚗", "motorcycle": "🏍️", "truck": "🚛"},
    },
}


# ─────────────────────────────────────────────
# FUNGSI HELPER KONFIGURASI
# ─────────────────────────────────────────────

def get_config(model_type: str) -> dict:
    """
    Ambil konfigurasi berdasarkan tipe model.

    Parameter:
        model_type : "original" atau "cctv"

    Return:
        dict konfigurasi model (path, classes, colors, emojis)
    """
    return MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["original"])


def get_model_path(model_type: str) -> str:
    """
    Ambil path file .pt berdasarkan tipe model.

    Parameter:
        model_type : "original" atau "cctv"

    Return:
        string path ke file .pt
    """
    config = get_config(model_type)
    return config["path"]


def get_available_model_types() -> list:
    """
    Kembalikan daftar tipe model yang tersedia.

    Return:
        list string nama model
    """
    return list(MODEL_CONFIGS.keys())
