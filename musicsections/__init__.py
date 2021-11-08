#!/usr/bin/env python
"""Top-level module for musicsections"""

from .version import version as __version__
from .deepsim import load_deepsim_model
from .fewshot import load_fewshot_model
from .core import segment_file
from .viz import plot_segmentation, plot_segmentation_json
