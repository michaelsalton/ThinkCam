#!/bin/bash
# Launcher for EVS capture script on OpenSUSE Leap 16
# Sets LD_LIBRARY_PATH to the extracted ArenaSDK libraries and activates the venv.

SDK=/home/charalambos/Downloads/ArenaSDK_Linux_x64
RDMA=/home/charalambos/Downloads/rdma_libs/usr/lib64

export LD_LIBRARY_PATH=\
$SDK/lib64:\
$SDK/OpenCV/lib:\
$SDK/GenICam/library/lib/Linux64_x64:\
$SDK/Metavision/lib:\
$SDK/ffmpeg:\
$RDMA:\
/tmp/usr/lib64

source ~/envs/default/bin/activate
python3 "$(dirname "$0")/evs_capture_visualize.py" "$@"
