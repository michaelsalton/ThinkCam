# ThinkCam

Real-time visualization and capture tool for the LUCID TRT009S-E event camera (EVS). Built with PySide6 and OpenCV. Streams CDFrame at the camera's native frame-generator FPS.

## Features

- Live CDFrame view (accumulated events rendered as grayscale)
- **Save PNG** snapshot
- **Record Video** to MP4
- Status bar with event rate, GVSP FPS, link bandwidth, render time, and frame counter

## Requirements

- Python 3.10+
- LUCID ArenaSDK for Linux x64 ([download](https://thinklucid.com/downloads-hub/))
- `arena_api` Python wheel (separate download from the same downloads hub)

### Python packages

```bash
pip install -r requirements.txt
pip install arena_api-*.whl
```

## Setup

1. **ArenaSDK**: Download and extract ArenaSDK for Linux x64. Set the `ARENA_SDK` environment variable to the extracted `ArenaSDK_Linux_x64` directory, or edit the default in `run_evs.sh`.

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

```bash
./run_evs.sh
```

### Keyboard shortcuts

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
    visualizer.py      # CDFrame buffer -> BGR
    controls.py        # Save / Record sidebar
    status_bar.py      # Live statistics status bar
    recorder.py        # MP4 video recording
    constants.py       # Camera defaults
  run_evs.sh           # Launcher script
  requirements.txt
```

## License

MIT
