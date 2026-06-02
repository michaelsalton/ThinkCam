import ctypes

import cv2
import numpy as np

from thinkcam.constants import DENOISE_KERNEL


def render_events(
    events: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, int, int]:
    """Render a batch of raw events to BGR and return (bgr, pos_count, neg_count).

    `events` is an (N, 4) float32 array of (x, y, t, p) as delivered by the
    camera in XYTPFrame mode. Positive polarity -> white, negative -> black,
    no event -> mid-gray. Unlike the old CDFrame path, the counts here are the
    real per-event tallies, not pixel thresholds.

    Polarity is treated as positive when p > 0, which handles both {0,1}
    (0 = neg, 1 = pos) and {-1,1} encodings without assuming which one the
    sensor emits (see docs/ImplementationPlan.md §2).
    """
    bgr = np.full((height, width, 3), 128, dtype=np.uint8)

    if events.shape[0] == 0:
        return bgr, 0, 0

    x = events[:, 0].astype(np.int32)
    y = events[:, 1].astype(np.int32)
    p = events[:, 3]

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, p = x[valid], y[valid], p[valid]

    pos = p > 0
    bgr[y[pos], x[pos]] = (255, 255, 255)
    bgr[y[~pos], x[~pos]] = (0, 0, 0)

    return bgr, int(np.count_nonzero(pos)), int(np.count_nonzero(~pos))


def render_cdframe(buffer) -> tuple[np.ndarray, int, int]:
    """Render a CDFrame to BGR and return (bgr, pos_count, neg_count).

    CDFrame encodes per-pixel ∂I/∂t as polarity: white = positive event,
    black = negative event, mid-gray = no change. Counts are taken from the
    raw mono buffer before denoising so the time-series reflects the camera's
    actual event output.
    """
    bpp = buffer.bits_per_pixel
    channels = max(1, bpp // 8)
    raw = (ctypes.c_ubyte * buffer.size_filled).from_address(
        ctypes.addressof(buffer.pbytes)
    )
    arr = np.frombuffer(raw, dtype=np.uint8)

    if channels == 3:
        img = arr.reshape((buffer.height, buffer.width, 3))
        mono = img[:, :, 0]
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        mono = arr[: buffer.height * buffer.width].reshape(
            (buffer.height, buffer.width)
        )
        bgr = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

    pos_count = int(np.count_nonzero(mono > 200))
    neg_count = int(np.count_nonzero(mono < 50))

    denoised = cv2.morphologyEx(bgr, cv2.MORPH_OPEN, DENOISE_KERNEL)
    return denoised, pos_count, neg_count
