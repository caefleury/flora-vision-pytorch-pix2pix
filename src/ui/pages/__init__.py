"""
UI Pages - Individual page renderers.
"""

from .analysis import render as render_analysis
from .training import render as render_training
from .comparison import render as render_comparison
from .about import render as render_about

__all__ = [
    "render_analysis",
    "render_training",
    "render_comparison",
    "render_about",
]
