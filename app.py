import io
import os
import tempfile
import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px

from app import (
    load_model,
    detect_vehicles,
    detect_vehicles_video,
    get_config,
    get_detection_summary,
    get_dominant_class,
    build_dataframe_data,
)

import io
import os
import tempfile
import streamlit as st
from PIL import Image

from app import (
    load_model,
    detect_vehicles,
    detect_vehicles_video,
    get_config,
    get_detection_summary,
    get_dominant_class,
    build_dataframe_data,
)


# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Vehicle Detection App",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# LOAD MODEL (cache agar tidak reload tiap interaksi)
# ─────────────────────────────────────────────

@st.cache_resource
def get_model(model_path: str):
    return load_model(model_path)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/car.png", width=80)
    st.title("⚙️ Pengaturan")
    st.markdown("---")

    # ── Pilih Model ──────────────────────────
    st.markdown("### 🤖 Pilih Model")
    model_choice = st.radio(
        label="Model",
        options=["Original (bus/car/van)", "CCTV (car/motorcycle/truck)"],
        index=0,
        help="Original = dataset umum | CCTV = dilatih dari video CCTV Cisarua"
    )
    model_type = "original" if "Original" in model_choice else "cctv"
    config     = get_config(model_type)

    st.markdown("---")

    # ── Threshold ────────────────────────────
    st.markdown("### 🎚️ Threshold")
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.1, max_value=0.95, value=0.5, step=0.05,
        help="Semakin tinggi = hanya deteksi yang yakin saja"
    )
    iou = st.slider(
        "IoU Threshold",
        min_value=0.1, max_value=0.95, value=0.45, step=0.05,
        help="Kontrol overlap antar bounding box"
    )

    st.markdown("---")

    # ── Keterangan Warna ─────────────────────
    st.markdown("### 🎨 Keterangan Warna")
    if model_type == "original":
        st.markdown("🟠 **Orange** → Bus")
        st.markdown("🟢 **Hijau** → Car")
        st.markdown("🟣 **Ungu** → Van")
    else:
        st.markdown("🟢 **Hijau** → Car")
        st.markdown("🔵 **Biru** → Motorcycle")
        st.markdown("🔴 **Merah** → Truck")

    st.markdown("---")
    st.markdown("### ℹ️ Tentang App")
    st.markdown(
        "Deteksi kendaraan menggunakan **YOLOv11**. "
        "Pilih model Original untuk gambar umum, "
        "atau CCTV untuk video traffic kamera."
    )


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

model_path = config["path"]
model      = get_model(model_path)

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = True
    st.success(f"✅ Model **{model_choice}** siap digunakan!")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("🚗 Vehicle Detection App")
st.markdown(
    f"Model aktif: **{model_choice}** | "
    f"Class: **{', '.join(config['classes'])}**"
)
st.markdown("---")


# ─────────────────────────────────────────────
# TAB: GAMBAR | VIDEO
# ─────────────────────────────────────────────

tab_image, tab_video = st.tabs(["📸 Deteksi Gambar", "🎥 Deteksi Video"])


# ═══════════════════════════════════════════
# TAB 1 — DETEKSI GAMBAR  ← TIDAK DIUBAH
# ═══════════════════════════════════════════

with tab_image:
    st.markdown("### 📥 Upload Gambar")
    uploaded_file = st.file_uploader(
        "Pilih gambar kendaraan (JPG / JPEG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="uploader_image"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🖼️ Gambar Input")
            st.image(image, use_container_width=True)
            st.caption(f"Ukuran: {image.size[0]} x {image.size[1]} px")

        st.markdown("---")
        detect_clicked = st.button(
            "🔍 Deteksi Kendaraan",
            type="primary",
            use_container_width=True,
            key="btn_detect_image"
        )

        if detect_clicked:
            with st.spinner("Sedang mendeteksi..."):
                result = detect_vehicles(
                    model         = model,
                    image         = image,
                    confidence    = confidence,
                    iou_threshold = iou,
                    model_type    = model_type
                )

            counts          = result["counts"]
            total           = result["total"]
            annotated_image = result["annotated_image"]
            detections      = result["detections"]

            with col2:
                st.markdown("#### 📤 Hasil Deteksi")
                st.image(annotated_image, use_container_width=True)

            st.markdown("---")
            st.markdown("### 📊 Hasil Deteksi")

            # ── Metric cards ──────────────────
            cols   = st.columns(len(counts) + 1)
            emojis = config["emojis"]
            for i, (cls, count) in enumerate(counts.items()):
                with cols[i]:
                    st.metric(
                        label=f"{emojis.get(cls, '🚘')} {cls.capitalize()}",
                        value=count
                    )
            with cols[-1]:
                st.metric(label="📦 Total", value=total)

            # ── Ringkasan ─────────────────────
            st.markdown("### 📋 Ringkasan")
            summary = get_detection_summary(counts)
            if total > 0:
                st.success(f"**Terdeteksi:** {summary}")
            else:
                st.warning(
                    "Tidak ada kendaraan terdeteksi. "
                    "Coba turunkan Confidence Threshold."
                )

            # ── Detail per objek ──────────────
            if detections:
                with st.expander(f"🔎 Detail {len(detections)} objek terdeteksi"):
                    for i, det in enumerate(detections, 1):
                        st.markdown(
                            f"**{i}.** `{det['class']}` — "
                            f"confidence: `{det['confidence']:.2f}` — "
                            f"bbox: `{det['bbox']}`"
                        )

            # ── Download gambar hasil ─────────
            buf = io.BytesIO()
            annotated_image.save(buf, format="JPEG")
            buf.seek(0)
            st.download_button(
                "⬇️ Download Gambar Hasil",
                data      = buf,
                file_name = "hasil_deteksi.jpg",
                mime      = "image/jpeg",
                use_container_width=True
            )
    else:
        st.info("👆 Upload gambar untuk memulai deteksi.")


# ═══════════════════════════════════════════
# TAB 2 — DETEKSI VIDEO  ← DIUBAH: tanpa counting & analytics
# ═══════════════════════════════════════════

with tab_video:
    st.markdown("### 📥 Upload Video")
    st.info(
        "💡 **Tips:** Gunakan model **CCTV** di sidebar untuk video traffic kamera."
    )

    uploaded_video = st.file_uploader(
        "Pilih video kendaraan (MP4 / AVI / MOV)",
        type=["mp4", "avi", "mov"],
        key="uploader_video"
    )

    if uploaded_video is not None:

        # ── Simpan video upload ke file temp ──
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()
        input_path = tfile.name

        # ── Preview video input ───────────────
        st.markdown("#### 🎬 Preview Video Input")
        st.video(uploaded_video)

        st.markdown("---")

        detect_video_clicked = st.button(
            "🔍 Mulai Deteksi Video",
            type="primary",
            use_container_width=True,
            key="btn_detect_video"
        )

        if detect_video_clicked:

            # ── Output path ───────────────────
            output_path = tempfile.mktemp(suffix="_result.mp4")

            # ── Progress bar ──────────────────
            st.markdown("#### ⏳ Proses Deteksi")
            progress_bar = st.progress(0)
            status_text  = st.empty()

            def update_progress(val):
                progress_bar.progress(min(val, 1.0))
                status_text.text(f"Memproses... {int(val * 100)}%")

            with st.spinner("Sedang memproses video..."):
                result = detect_vehicles_video(
                    model             = model,
                    video_path        = input_path,
                    output_path       = output_path,
                    confidence        = confidence,
                    iou_threshold     = iou,
                    model_type        = model_type,
                    progress_callback = update_progress
                )

            progress_bar.progress(1.0)
            status_text.text("✅ Selesai!")

            st.markdown("---")

            # ── Info ringkas ──────────────────
            st.markdown("### 📊 Info Deteksi")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("⚡ FPS Rata-rata", result["fps_avg"])
            with col2:
                st.metric("🎞️ Total Frame Diproses", result["frame_count"])

            st.success(
                f"✅ Deteksi selesai! Kendaraan yang dideteksi: "
                f"**{', '.join(config['classes'])}**"
            )

            st.markdown("---")

            # ── Download & preview hasil ──────
            st.markdown("### 💾 Download Hasil")
            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    video_bytes = f.read()

                st.download_button(
                    label       = "⬇️ Download Video Hasil Deteksi",
                    data        = video_bytes,
                    file_name   = "hasil_deteksi.mp4",
                    mime        = "video/mp4",
                    use_container_width=True
                )

                st.markdown("#### 🎬 Preview Hasil Deteksi")
                st.video(video_bytes)

            # ── Cleanup temp files ────────────
            try:
                os.unlink(input_path)
                os.unlink(output_path)
            except Exception:
                pass

    else:
        st.info("👆 Upload video untuk memulai deteksi.")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Gunakan model **CCTV** di sidebar untuk hasil terbaik
        - Format video yang didukung: MP4, AVI, MOV
        - Semakin panjang video → semakin lama proses
        """)

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Vehicle Detection App",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# LOAD MODEL (cache agar tidak reload tiap interaksi)
# ─────────────────────────────────────────────

@st.cache_resource
def get_model(model_path: str):
    return load_model(model_path)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/car.png", width=80)
    st.title("⚙️ Pengaturan")
    st.markdown("---")

    # ── Pilih Model ──────────────────────────
    st.markdown("### 🤖 Pilih Model")
    model_choice = st.radio(
        label="Model",
        options=["Original (bus/car/van)", "CCTV (car/motorcycle/truck)"],
        index=0,
        help="Original = dataset umum | CCTV = dilatih dari video CCTV Cisarua"
    )
    model_type = "original" if "Original" in model_choice else "cctv"
    config     = get_config(model_type)

    st.markdown("---")

    # ── Threshold ────────────────────────────
    st.markdown("### 🎚️ Threshold")
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.1, max_value=0.95, value=0.5, step=0.05,
        help="Semakin tinggi = hanya deteksi yang yakin saja"
    )
    iou = st.slider(
        "IoU Threshold",
        min_value=0.1, max_value=0.95, value=0.45, step=0.05,
        help="Kontrol overlap antar bounding box"
    )

    st.markdown("---")

    # ── Pengaturan Garis (khusus video) ──────
    st.markdown("### 📏 Garis Counting (Video)")
    line_position = st.slider(
        "Posisi Garis",
        min_value=0.2, max_value=0.8, value=0.5, step=0.05,
        help="0.5 = tengah layar | Geser sesuai posisi jalan di video"
    )

    st.markdown("---")

    # ── Keterangan Warna ─────────────────────
    st.markdown("### 🎨 Keterangan Warna")
    if model_type == "original":
        st.markdown("🟠 **Orange** → Bus")
        st.markdown("🟢 **Hijau** → Car")
        st.markdown("🟣 **Ungu** → Van")
    else:
        st.markdown("🟢 **Hijau** → Car")
        st.markdown("🔵 **Biru** → Motorcycle")
        st.markdown("🔴 **Merah** → Truck")

    st.markdown("---")
    st.markdown("### ℹ️ Tentang App")
    st.markdown(
        "Deteksi kendaraan menggunakan **YOLOv11**. "
        "Pilih model Original untuk gambar umum, "
        "atau CCTV untuk video traffic kamera."
    )


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

model_path = config["path"]
model      = get_model(model_path)

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = True
    st.success(f"✅ Model **{model_choice}** siap digunakan!")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("🚗 Vehicle Detection App")
st.markdown(
    f"Model aktif: **{model_choice}** | "
    f"Class: **{', '.join(config['classes'])}**"
)
st.markdown("---")


# ─────────────────────────────────────────────
# TAB: GAMBAR | VIDEO
# ─────────────────────────────────────────────

tab_image, tab_video = st.tabs(["📸 Deteksi Gambar", "🎥 Deteksi Video"])


# ═══════════════════════════════════════════
# TAB 1 — DETEKSI GAMBAR
# ═══════════════════════════════════════════

with tab_image:
    st.markdown("### 📥 Upload Gambar")
    uploaded_file = st.file_uploader(
        "Pilih gambar kendaraan (JPG / JPEG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="uploader_image"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🖼️ Gambar Input")
            st.image(image, use_container_width=True)
            st.caption(f"Ukuran: {image.size[0]} x {image.size[1]} px")

        st.markdown("---")
        detect_clicked = st.button(
            "🔍 Deteksi Kendaraan",
            type="primary",
            use_container_width=True,
            key="btn_detect_image"
        )

        if detect_clicked:
            with st.spinner("Sedang mendeteksi..."):
                result = detect_vehicles(
                    model         = model,
                    image         = image,
                    confidence    = confidence,
                    iou_threshold = iou,
                    model_type    = model_type
                )

            counts          = result["counts"]
            total           = result["total"]
            annotated_image = result["annotated_image"]
            detections      = result["detections"]

            with col2:
                st.markdown("#### 📤 Hasil Deteksi")
                st.image(annotated_image, use_container_width=True)

            st.markdown("---")
            st.markdown("### 📊 Hasil Deteksi")

            # ── Metric cards ──────────────────
            cols   = st.columns(len(counts) + 1)
            emojis = config["emojis"]
            for i, (cls, count) in enumerate(counts.items()):
                with cols[i]:
                    st.metric(
                        label=f"{emojis.get(cls, '🚘')} {cls.capitalize()}",
                        value=count
                    )
            with cols[-1]:
                st.metric(label="📦 Total", value=total)

            # ── Ringkasan ─────────────────────
            st.markdown("### 📋 Ringkasan")
            summary = get_detection_summary(counts)
            if total > 0:
                st.success(f"**Terdeteksi:** {summary}")
            else:
                st.warning(
                    "Tidak ada kendaraan terdeteksi. "
                    "Coba turunkan Confidence Threshold."
                )

            # ── Detail per objek ──────────────
            if detections:
                with st.expander(f"🔎 Detail {len(detections)} objek terdeteksi"):
                    for i, det in enumerate(detections, 1):
                        st.markdown(
                            f"**{i}.** `{det['class']}` — "
                            f"confidence: `{det['confidence']:.2f}` — "
                            f"bbox: `{det['bbox']}`"
                        )

            # ── Download gambar hasil ─────────
            buf = io.BytesIO()
            annotated_image.save(buf, format="JPEG")
            buf.seek(0)
            st.download_button(
                "⬇️ Download Gambar Hasil",
                data      = buf,
                file_name = "hasil_deteksi.jpg",
                mime      = "image/jpeg",
                use_container_width=True
            )
    else:
        st.info("👆 Upload gambar untuk memulai deteksi.")


# ═══════════════════════════════════════════
# TAB 2 — DETEKSI VIDEO
# ═══════════════════════════════════════════

with tab_video:
    st.markdown("### 📥 Upload Video")
    st.info(
        "💡 **Tips:** Gunakan model **CCTV** di sidebar untuk video traffic kamera. "
        "Atur posisi **Garis Counting** sesuai lokasi jalan di video."
    )

    uploaded_video = st.file_uploader(
        "Pilih video kendaraan (MP4 / AVI / MOV)",
        type=["mp4", "avi", "mov"],
        key="uploader_video"
    )

    if uploaded_video is not None:

        # ── Simpan video upload ke file temp ──
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()
        input_path = tfile.name

        # ── Preview video input ───────────────
        st.markdown("#### 🎬 Preview Video Input")
        st.video(uploaded_video)

        st.markdown("---")

        detect_video_clicked = st.button(
            "🔍 Mulai Deteksi Video",
            type="primary",
            use_container_width=True,
            key="btn_detect_video"
        )

        if detect_video_clicked:

            # ── Output path ───────────────────
            output_path = tempfile.mktemp(suffix="_result.mp4")

            # ── Progress bar ──────────────────
            st.markdown("#### ⏳ Proses Deteksi")
            progress_bar = st.progress(0)
            status_text  = st.empty()

            def update_progress(val):
                progress_bar.progress(min(val, 1.0))
                status_text.text(f"Memproses... {int(val * 100)}%")

            with st.spinner("Sedang memproses video..."):
                result = detect_vehicles_video(
                    model             = model,
                    video_path        = input_path,
                    output_path       = output_path,
                    confidence        = confidence,
                    iou_threshold     = iou,
                    model_type        = model_type,
                    line_position     = line_position,
                    progress_callback = update_progress
                )

            progress_bar.progress(1.0)
            status_text.text("✅ Selesai!")

            counts  = result["total_counts"]
            total   = result["total"]
            fps_avg = result["fps_avg"]

            st.markdown("---")

            # ── Metric cards ──────────────────
            st.markdown("### 📊 Hasil Vehicle Counting")
            emojis = config["emojis"]

            cols = st.columns(len(counts) + 2)
            for i, (cls, count) in enumerate(counts.items()):
                with cols[i]:
                    st.metric(f"{emojis.get(cls, '🚘')} {cls.capitalize()}", count)
            with cols[-2]:
                st.metric("📦 Total", total)
            with cols[-1]:
                st.metric("⚡ FPS Rata-rata", fps_avg)

            st.markdown("---")

            # ── Analytics ─────────────────────
            st.markdown("### 📈 Analytics")

            if total > 0:
                df_data = build_dataframe_data(counts, total)
                df      = pd.DataFrame(df_data)

                col_bar, col_pie = st.columns(2)

                with col_bar:
                    st.markdown("#### 📊 Bar Chart")
                    fig_bar = px.bar(
                        df,
                        x     = "Kelas",
                        y     = "Jumlah",
                        color = "Kelas",
                        text  = "Jumlah",
                        title = "Jumlah Kendaraan per Kelas",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig_bar.update_traces(textposition="outside")
                    fig_bar.update_layout(showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)

                with col_pie:
                    st.markdown("#### 🥧 Pie Chart")
                    fig_pie = px.pie(
                        df,
                        names  = "Kelas",
                        values = "Jumlah",
                        title  = "Persentase Kendaraan",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                # ── Tabel detail ──────────────
                st.markdown("#### 📋 Tabel Detail")
                st.dataframe(df, use_container_width=True, hide_index=True)

                # ── Kelas dominan ─────────────
                dominan, dominan_count = get_dominant_class(counts)
                if dominan:
                    pct = round(dominan_count / total * 100, 1)
                    st.success(
                        f"🏆 Kendaraan dominan: **{dominan.capitalize()}** "
                        f"({dominan_count} kendaraan, {pct}%)"
                    )
            else:
                st.warning(
                    "Tidak ada kendaraan terdeteksi. "
                    "Coba turunkan Confidence Threshold."
                )

            st.markdown("---")

            # ── Download video hasil ──────────
            st.markdown("### 💾 Download Hasil")
            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    video_bytes = f.read()

                st.download_button(
                    label       = "⬇️ Download Video Hasil Deteksi",
                    data        = video_bytes,
                    file_name   = "hasil_deteksi_cctv.mp4",
                    mime        = "video/mp4",
                    use_container_width=True
                )

                # ── Preview hasil ─────────────
                st.markdown("#### 🎬 Preview Hasil Deteksi")
                st.video(video_bytes)

            # ── Cleanup temp files ────────────
            try:
                os.unlink(input_path)
                os.unlink(output_path)
            except Exception:
                pass

    else:
        st.info("👆 Upload video untuk memulai deteksi.")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Gunakan model **CCTV** di sidebar untuk hasil terbaik
        - Atur **Posisi Garis** sesuai lokasi jalan di video kamu
        - Format video yang didukung: MP4, AVI, MOV
        - Semakin panjang video → semakin lama proses
        """)
