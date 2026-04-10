# ThinkCam

Real-time visualization and capture tool for the LUCID TRT009S-E event camera (EVS). Built with PySide6, OpenCV, and dv-processing.

## Features

- **5 visualization modes** cycled from the sidebar or keyboard:
  - **CDFrame** -- accumulated event frames as grayscale
  - **XYTPFrame** -- per-event polarity heatmap (blue = ON, red = OFF)
  - **DV-Events** -- dv-processing EventVisualizer with colour-coded polarity
  - **DV-Accumulator** -- accumulated potential surface showing event density
  - **DV-TimeSurface** -- time-surface with selectable colourmap (recent events = warm)
- **Colourmap picker** -- JET, VIRIDIS, MAGMA, INFERNO, TURBO, PLASMA, HOT, BONE
- **Noise filtering** -- hardware bias controls (threshold, refractory, burst filter) and dv-processing BackgroundActivityNoiseFilter with adjustable duration
- **Export** -- save PNG snapshots or record MP4 video
- **Live stats** -- event rate, GVS frame rate, link throughput, render time

## Requirements

- Python 3.10+
- LUCID ArenaSDK for Linux x64 ([download](https://thinklucid.com/downloads-hub/))
- `arena_api` Python wheel (bundled with ArenaSDK download)

### System libraries (openSUSE)

```bash
sudo zypper install libgthread-2_0-0 libibverbs librdmacm1
```

### Python packages

```bash
pip install -r requirements.txt
pip install arena_api-2.8.4-py3-none-any.whl  # from ArenaSDK download
```

## Setup

1. **ArenaSDK**: Download and extract ArenaSDK for Linux x64 from [LUCID Downloads](https://thinklucid.com/downloads-hub/). Set the `ARENA_SDK` environment variable to the extracted path, or place it at `~/ArenaSDK_Linux_x64`.

2. **arena_api config**: Point the Python wrapper at your SDK's native libraries by editing `arena_api_config.py` in your site-packages:

   ```python
   ARENAC_CUSTOM_PATHS = {
       ...
       'python64_lin': '/path/to/ArenaSDK_Linux_x64/lib64/libarenac.so'
   }
   SAVEC_CUSTOM_PATHS = {
       ...
       'python64_lin': '/path/to/ArenaSDK_Linux_x64/lib64/libsavec.so'
   }
   ```

3. **Network**: The camera uses link-local addressing. Assign an IP on the same subnet to your Ethernet interface:

   ```bash
   sudo ip addr add 169.254.80.1/16 dev <interface>
   ```

   Default camera IP is `169.254.80.199` (configurable in `thinkcam/constants.py`).

## Usage

### GUI (default)

```bash
./run_evs.sh
```

### CLI (original OpenCV viewer)

```bash
./run_evs.sh --cli
```

### Keyboard shortcuts (GUI)

| Key | Action |
|-----|--------|
| S | Save PNG snapshot |
| Q / Esc | Quit |

## Project structure

```
ThinkCam/
  thinkcam/
    main.py            # Application entry point
    main_window.py     # Main window layout and signal wiring
    camera_worker.py   # QThread for camera acquisition
    visualizer.py      # Frame rendering (5 modes)
    controls.py        # Sidebar control panel
    status_bar.py      # Live statistics status bar
    recorder.py        # MP4 video recording
    constants.py       # Shared configuration and defaults
  evs_capture_visualize.py   # Standalone CLI viewer (legacy)
  run_evs.sh                 # Launcher script
  requirements.txt
```

## License

MIT
