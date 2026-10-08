#!/usr/bin/env bash
# Run Tank stereo-inertial TagFusion (left/right + imu0.txt).
#
# 用法:
#   ./run_stereo_inertial_test.sh
#   ./run_stereo_inertial_test.sh /path/to/short_test [traj_name]
#
# 环境变量（均可选）: VOC / SETTINGS / BIN / SEQ
#
# IMU 默认读 <seq>/imu0.txt；yaml 中 IMU.Frequency 须与实测采样率一致（本数据约 333 Hz）。
# 若尚无 imu0.txt:
#   python3 extract_imu_from_bag.py /home/geneta/下载/Tank/rosbag

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORB_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

VOC="${VOC:-${ORB_ROOT}/Vocabulary/ORBvoc.txt}"
SETTINGS="${SETTINGS:-${SCRIPT_DIR}/test_stereo+imu.yaml}"
BIN="${BIN:-${SCRIPT_DIR}/stereo_inertial_test_tag}"

DEFAULT_SEQ="/home/geneta/下载/Tank/short_test"
SEQ_PATH="${1:-${SEQ:-${DEFAULT_SEQ}}}"
TRAJ_NAME="${2:-short_test_stereo_inertial}"

if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: binary not found: ${BIN}"
  echo "Build first, e.g.:"
  echo "  cmake --build ${ORB_ROOT}/build --target stereo_inertial_test_tag -j"
  exit 1
fi

if [[ ! -f "${VOC}" ]]; then
  echo "ERROR: vocabulary not found: ${VOC}"
  exit 1
fi

if [[ ! -f "${SETTINGS}" ]]; then
  echo "ERROR: settings not found: ${SETTINGS}"
  exit 1
fi

if [[ ! -d "${SEQ_PATH}" ]]; then
  echo "ERROR: sequence path not found: ${SEQ_PATH}"
  exit 1
fi

if [[ ! -f "${SEQ_PATH}/imu0.txt" ]]; then
  echo "ERROR: ${SEQ_PATH}/imu0.txt not found"
  echo "Extract IMU first:"
  echo "  python3 ${SCRIPT_DIR}/extract_imu_from_bag.py /home/geneta/下载/Tank/rosbag"
  exit 1
fi

echo "VOC      = ${VOC}"
echo "SETTINGS = ${SETTINGS}"
echo "SEQ      = ${SEQ_PATH}"
echo "TRAJ     = ${TRAJ_NAME}"
echo "BIN      = ${BIN}"
echo "-------"

cd "${SCRIPT_DIR}"
exec "${BIN}" "${VOC}" "${SETTINGS}" "${SEQ_PATH}" "${TRAJ_NAME}"
