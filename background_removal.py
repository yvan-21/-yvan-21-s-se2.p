"""Shared foreground extraction used by previews and dataset preprocessing."""

import logging

import cv2
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


def remove_background(image: Image.Image) -> Image.Image:
    """Isolate the centered foreground subject using seeded GrabCut.

    A definite foreground seed in the center and background seeds on the edges
    work better for close-up leaf photos than relying on green color alone.
    """
    source_image = image.convert("RGB")
    image_array = np.asarray(source_image)
    height, width = image_array.shape[:2]

    scale = min(800 / max(height, width), 1.0)
    if scale < 1.0:
        small_image = cv2.resize(
            image_array,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small_image = image_array

    small_height, small_width = small_image.shape[:2]
    border_x = max(2, int(small_width * 0.04))
    border_y = max(2, int(small_height * 0.04))
    mask = np.full(
        (small_height, small_width),
        cv2.GC_PR_FGD,
        dtype=np.uint8,
    )
    mask[:border_y, :] = cv2.GC_BGD
    mask[-border_y:, :] = cv2.GC_BGD
    mask[:, :border_x] = cv2.GC_BGD
    mask[:, -border_x:] = cv2.GC_BGD

    center_x = small_width // 2
    center_y = small_height // 2
    seed_half_width = max(4, int(small_width * 0.16))
    seed_half_height = max(4, int(small_height * 0.16))
    mask[
        max(border_y, center_y - seed_half_height):min(small_height - border_y, center_y + seed_half_height),
        max(border_x, center_x - seed_half_width):min(small_width - border_x, center_x + seed_half_width),
    ] = cv2.GC_FGD

    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        cv2.cvtColor(small_image, cv2.COLOR_RGB2BGR),
        mask,
        None,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_MASK,
    )

    foreground_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    component_count, components, stats, _ = cv2.connectedComponentsWithStats(
        foreground_mask,
        connectivity=8,
    )
    if component_count > 1:
        center_component = int(components[center_y, center_x])
        largest_component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        selected_component = (
            center_component if center_component > 0 else largest_component
        )
        foreground_mask = np.where(
            components == selected_component,
            255,
            0,
        ).astype(np.uint8)

    foreground_fraction = np.count_nonzero(foreground_mask) / foreground_mask.size
    if foreground_fraction < 0.01 or foreground_fraction > 0.98:
        logger.warning(
            "Background removal produced an unreliable mask (%.2f%% foreground); preserving source image",
            foreground_fraction * 100,
        )
        return source_image

    if scale < 1.0:
        foreground_mask = cv2.resize(
            foreground_mask,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
    foreground_mask = cv2.GaussianBlur(foreground_mask, (3, 3), 0)
    result = cv2.bitwise_and(image_array, image_array, mask=foreground_mask)
    return Image.fromarray(result)
