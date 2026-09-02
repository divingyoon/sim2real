#!/bin/bash
source "$(dirname "$0")/common.sh"
pkill -f realsense2_camera_node || true
pkill -f "ros2 launch realsense2_camera" || true
echo "camera down"
