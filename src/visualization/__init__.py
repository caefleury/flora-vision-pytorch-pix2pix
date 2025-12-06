"""
Visualization module - Heatmaps, Grad-CAM and plots.
"""

from .heatmaps import create_heatmap_figure, create_simple_overlay
from .gradcam import (
    ColorModelGradCAM,
    get_available_cam_methods,
    get_cam_method_description,
    apply_colormap,
    overlay_cam_on_image,
)

__all__ = [
    "create_heatmap_figure",
    "create_simple_overlay",
    "ColorModelGradCAM",
    "get_available_cam_methods",
    "get_cam_method_description",
    "apply_colormap",
    "overlay_cam_on_image",
]
