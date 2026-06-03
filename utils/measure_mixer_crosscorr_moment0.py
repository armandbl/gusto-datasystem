#!/usr/bin/env python3
"""Backward-compatibility shim for the moment-0 cross-correlation workflow.

All functionality now lives in ``measure_mixer_crosscorr``.
"""

from measure_mixer_crosscorr import build_moment0_map, load_cube, moment0_header, main  # noqa: F401
