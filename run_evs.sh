#!/bin/bash
# Launcher for EVS capture script on OpenSUSE Leap 16
# Sets LD_LIBRARY_PATH to the extracted ArenaSDK libraries and activates the venv.

SDK=${ARENA_SDK:-$HOME/Downloads/ArenaViewMP_v_1.0.0.10_Linux_x64/ArenaSDK_Linux_x64}
RDMA=${RDMA_LIBS:-$HOME/rdma_libs/usr/lib64}

if [ ! -d "$SDK" ]; then
    echo "ERROR: ArenaSDK not found at $SDK"
    echo "Download it from https://thinklucid.com/downloads-hub/"
    echo "Extract it and either place it at ~/ArenaSDK_Linux_x64"
    echo "or set ARENA_SDK=/path/to/ArenaSDK_Linux_x64"
    exit 1
fi

export LD_LIBRARY_PATH=\
$SDK/lib64:\
$SDK/OpenCV/lib:\
$SDK/GenICam/library/lib/Linux64_x64:\
$SDK/Metavision/lib:\
$SDK/ffmpeg:\
$RDMA:\
/tmp/usr/lib64

source ~/envs/default/bin/activate

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$1" = "--cli" ]; then
    shift
    python3 "$SCRIPT_DIR/evs_capture_visualize.py" "$@"
else
    python3 -m thinkcam.main "$@"
fi
