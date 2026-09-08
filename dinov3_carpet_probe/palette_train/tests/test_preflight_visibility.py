from __future__ import annotations

import numpy as np

from dinov3_carpet_probe.palette_train.src.preflight_analysis import _min_delta_to_pixels


def test_visibility_delta_is_zero_for_exact_pixel():
    image = np.array(
        [
            [[0, 0, 0], [255, 255, 255]],
            [[10, 20, 30], [40, 50, 60]],
        ],
        dtype=np.uint8,
    )
    assert _min_delta_to_pixels([10, 20, 30], image) == 0.0
