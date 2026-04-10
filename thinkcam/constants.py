import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Camera defaults
# ---------------------------------------------------------------------------
CAMERA_IP = "169.254.80.199"
NUM_BUFFERS = 10
IMAGE_TIMEOUT_MS = 2000
ERC_RATE_LIMIT_MEV = 10.0

# ---------------------------------------------------------------------------
# Visualization modes
# ---------------------------------------------------------------------------
MODE_CDFRAME = "CDFrame"
MODE_XYTPFRAME = "XYTPFrame"
MODE_DV_EVENTS = "DV-Events"
MODE_DV_ACCUMULATOR = "DV-Accumulator"
MODE_DV_TIMESURFACE = "DV-TimeSurface"

ALL_MODES = [
    MODE_CDFRAME,
    MODE_XYTPFRAME,
    MODE_DV_EVENTS,
    MODE_DV_ACCUMULATOR,
    MODE_DV_TIMESURFACE,
]

# Modes that need XYTPFrame output from the camera
XYTP_MODES = {MODE_XYTPFRAME, MODE_DV_EVENTS, MODE_DV_ACCUMULATOR, MODE_DV_TIMESURFACE}

# ---------------------------------------------------------------------------
# Colormaps (display name -> OpenCV constant)
# ---------------------------------------------------------------------------
COLORMAP_OPTIONS = {
    "JET": cv2.COLORMAP_JET,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "MAGMA": cv2.COLORMAP_MAGMA,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "TURBO": cv2.COLORMAP_TURBO,
    "PLASMA": cv2.COLORMAP_PLASMA,
    "HOT": cv2.COLORMAP_HOT,
    "BONE": cv2.COLORMAP_BONE,
}

# ---------------------------------------------------------------------------
# Hardware noise-filter defaults
# ---------------------------------------------------------------------------
BIAS_THRESHOLD_POS = 10
BIAS_THRESHOLD_NEG = 10
BIAS_REFRACTORY = 10
BURST_FILTER_ENABLE = True

# ---------------------------------------------------------------------------
# Software post-processing kernels
# ---------------------------------------------------------------------------
DENOISE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

# ---------------------------------------------------------------------------
# dv-processing defaults
# ---------------------------------------------------------------------------
DV_NOISE_DURATION_US = 2000
