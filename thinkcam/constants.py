import cv2

CAMERA_IP = "169.254.80.199"
NUM_BUFFERS = 10
IMAGE_TIMEOUT_MS = 2000
# Rate at which decoded frames are rendered and pushed to the GUI. The camera
# delivers event buffers far faster than this (~1.7k/s); emitting one Qt signal
# per buffer floods the GUI thread's queue and makes the display lag further and
# further behind real time. We consume every buffer (raw recording stays
# lossless) but accumulate events and only render/emit at this rate.
DISPLAY_FPS = 30.0
ERC_RATE_LIMIT_MEV = 10.0

BIAS_THRESHOLD_POS = 10
BIAS_THRESHOLD_NEG = 10
BIAS_REFRACTORY = 10
BURST_FILTER_ENABLE = True

DENOISE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# EVS stream output format. "XYTPFrame" delivers the raw asynchronous event
# stream as 4x float32 (x, y, t, p) per fired pixel — the lossless input the
# raw recorder taps. ("CDFrame" was the old preview-only rasterized mode.)
EVS_OUTPUT_FORMAT = "XYTPFrame"

# Raw event recorder
RAW_RECORD_DIR = "recordings"
# Bounded handoff queue between the acquisition QThread and the writer thread,
# measured in event *batches* (one per camera buffer). If it ever saturates the
# dropped-events counter rises (it must stay 0 at normal rates). The writer
# coalesces all queued batches into one HDF5 append, so this mainly needs to be
# deep enough to absorb a motion burst while the writer catches up; at ~1.7k
# buffers/s, 4096 is ~2.4s of headroom.
RAW_QUEUE_MAXSIZE = 4096
# HDF5 chunk length along the event axis.
RAW_HDF5_CHUNK = 1_000_000
# HDF5 compression for the event datasets. LZF caps the writer at ~8 Mev/s on
# this machine — below the camera's 10-MEV ErcRateLimit, so a fast scene
# overflows the queue and drops events. None lifts the write ceiling to
# ~50 Mev/s (well above any rate the sensor can emit) at the cost of larger
# files (~13 bytes/event vs ~8.4 compressed). Lossless capture wins; set to
# "lzf" or "gzip" only for long takes where disk space matters more than rate.
RAW_HDF5_COMPRESSION = None
