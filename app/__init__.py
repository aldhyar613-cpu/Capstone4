"""
app/__init__.py — Public API Package app/
=========================================
Import semua fungsi utama dari sini agar app.py cukup menulis:

    from app import load_model, detect_vehicles, ...

Atau bisa juga import langsung dari submodulnya:

    from app.detector import load_model
    from app.config   import get_config
    from app.utils    import get_detection_summary
"""

from app.config import (
    get_config,
    get_model_path,
    get_available_model_types,
    MODEL_CONFIGS,
)

from app.detector import (
    load_model,
    detect_vehicles,
    detect_vehicles_video,
    draw_boxes,
)

from app.utils import (
    get_detection_summary,
    format_result_text,
    get_dominant_class,
    calculate_percentage,
    build_dataframe_data,
)

__all__ = [
    # config
    "get_config",
    "get_model_path",
    "get_available_model_types",
    "MODEL_CONFIGS",
    # detector
    "load_model",
    "detect_vehicles",
    "detect_vehicles_video",
    "draw_boxes",
    # utils
    "get_detection_summary",
    "format_result_text",
    "get_dominant_class",
    "calculate_percentage",
    "build_dataframe_data",
]
