import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from background_removal import remove_background


class BackgroundRemovalTest(unittest.TestCase):
    def test_empty_grabcut_mask_preserves_source_instead_of_returning_black(self):
        source = Image.new("RGB", (224, 224), (120, 180, 90))

        def mark_every_pixel_as_background(*args):
            args[1].fill(cv2.GC_BGD)

        with patch(
            "background_removal.cv2.grabCut",
            side_effect=mark_every_pixel_as_background,
        ):
            result = remove_background(source)

        np.testing.assert_array_equal(np.asarray(result), np.asarray(source))

    def test_seeded_extraction_keeps_center_leaf_and_removes_border_background(self):
        size = 160
        source_array = np.full((size, size, 3), (80, 90, 100), dtype=np.uint8)
        cv2.ellipse(
            source_array,
            (size // 2, size // 2),
            (25, 60),
            0,
            0,
            360,
            (80, 170, 50),
            -1,
        )

        result = np.asarray(remove_background(Image.fromarray(source_array)))

        self.assertTrue(np.any(result[size // 2, size // 2] != 0))
        self.assertFalse(np.any(result[0, 0] != 0))
        self.assertLess(np.count_nonzero(np.any(result != 0, axis=2)), size * size * 0.5)


if __name__ == "__main__":
    unittest.main()
