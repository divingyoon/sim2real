#!/bin/bash
# 물체 하나의 FP++ 컨테이너 기동. usage: fpp_up.sh <name> <yaml_host_path>
# 이미지는 baked 라 패치 파일은 파일 단위 바인드 마운트(09.02 규약). 이름 fpp_<name>.
source "$(dirname "$0")/common.sh"
NAME=${1:?name}; YAML=${2:?yaml path}
[ -f "$YAML" ] || { echo "yaml missing: $YAML" >&2; exit 1; }
docker rm -f "fpp_$NAME" >/dev/null 2>&1 || true
docker run -d --name "fpp_$NAME" --network host --ipc=host --gpus all -e ROS_DOMAIN_ID=126 \
  -v $PPP/perception_plus_plus_core/detection/yolo.py:/workspace/perception_plus_plus/perception_plus_plus_core/detection/yolo.py:ro \
  -v $PPP/perception_plus_plus_core/fp_adapter/foundationpose_plus_plus.py:/workspace/perception_plus_plus/perception_plus_plus_core/fp_adapter/foundationpose_plus_plus.py:ro \
  -v $PPP/ros_ws/src/perception_plus_plus_ros/perception_plus_plus_ros/node.py:/opt/perception_plus_plus/lib/python3.10/site-packages/perception_plus_plus_ros/node.py:ro \
  -v $PPP/assets/meshes:/workspace/perception_plus_plus/assets/meshes:ro \
  -v "$YAML":/opt/params/"$NAME".yaml:ro \
  perception-plus-plus:humble-cup bash -lc "
    source /opt/ros/humble/setup.bash
    source /opt/perception_plus_plus/setup.bash
    cd /workspace/perception_plus_plus
    exec ros2 launch perception_plus_plus_ros cup_tracking.launch.py parameters_file:=/opt/params/$NAME.yaml" \
  >/dev/null
echo "fpp_$NAME up"
