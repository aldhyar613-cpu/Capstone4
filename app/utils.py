"""
app/utils.py — Fungsi Utilitas & Format Output
===============================================
Kumpulan fungsi bantu untuk memformat hasil deteksi.
Tidak bergantung pada Streamlit maupun model YOLO — murni helper.
"""


# ─────────────────────────────────────────────
# FORMAT OUTPUT
# ─────────────────────────────────────────────

def get_detection_summary(counts: dict) -> str:
    """
    Ringkasan singkat hasil deteksi dalam satu baris teks.

    Parameter:
        counts : dict {nama_class: jumlah_deteksi}

    Return:
        string ringkasan, contoh: "car 3, motorcycle 1"
        atau "Tidak ada kendaraan terdeteksi" jika semua nol
    """
    parts = [f"{cls} {count}" for cls, count in counts.items() if count > 0]
    return ", ".join(parts) if parts else "Tidak ada kendaraan terdeteksi"


def format_result_text(counts: dict, total: int) -> str:
    """
    Format hasil deteksi menjadi teks rapi multi-baris.

    Parameter:
        counts : dict {nama_class: jumlah_deteksi}
        total  : total seluruh deteksi

    Return:
        string multi-baris berformat tabel sederhana
    """
    lines = []
    for cls, count in counts.items():
        lines.append(f"{cls:<12}: {count}")
    lines.append("─" * 20)
    lines.append(f"Total       : {total} kendaraan")
    return "\n".join(lines)


def get_dominant_class(counts: dict) -> tuple:
    """
    Temukan kelas kendaraan yang paling banyak terdeteksi.

    Parameter:
        counts : dict {nama_class: jumlah_deteksi}

    Return:
        tuple (nama_class, jumlah) dari kelas dominan
        atau (None, 0) jika semua nol
    """
    if not counts or all(v == 0 for v in counts.values()):
        return (None, 0)
    dominan = max(counts, key=counts.get)
    return (dominan, counts[dominan])


def calculate_percentage(counts: dict, total: int) -> dict:
    """
    Hitung persentase tiap kelas dari total deteksi.

    Parameter:
        counts : dict {nama_class: jumlah_deteksi}
        total  : total seluruh deteksi

    Return:
        dict {nama_class: persentase_float}
        contoh: {"car": 66.7, "motorcycle": 33.3}
    """
    if total == 0:
        return {cls: 0.0 for cls in counts}
    return {cls: round(count / total * 100, 1) for cls, count in counts.items()}


def build_dataframe_data(counts: dict, total: int) -> dict:
    """
    Bangun data siap pakai untuk membuat DataFrame pandas.

    Parameter:
        counts : dict {nama_class: jumlah_deteksi}
        total  : total seluruh deteksi

    Return:
        dict dengan key "Kelas", "Jumlah", "Persentase"
        siap dipakai: pd.DataFrame(build_dataframe_data(counts, total))
    """
    percentages = calculate_percentage(counts, total)
    return {
        "Kelas"      : list(counts.keys()),
        "Jumlah"     : list(counts.values()),
        "Persentase" : [percentages[cls] for cls in counts],
    }
