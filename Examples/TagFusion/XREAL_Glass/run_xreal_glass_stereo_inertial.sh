#!/usr/bin/env bash
# 启动 XREAL Glass cam0+cam1+imu0 TagFusion（双目+IMU）
#
# 用法:
#   ./run_xreal_glass_stereo_inertial.sh
#   ./run_xreal_glass_stereo_inertial.sh /path/to/glass [traj_name]
#   ./run_xreal_glass_stereo_inertial.sh /path/to/seq_root [traj_name]
#   SEQ=/path/to/.../raw_data/glass ./run_xreal_glass_stereo_inertial.sh
#
# 环境变量（均可选）:
#   VOC / SETTINGS / BIN / SEQ
#
# IMU 默认读 glass/imu0.txt；yaml 中 IMU.Frequency 须与实测采样率一致（本数据约 1000 Hz）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORB_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

VOC="${VOC:-${ORB_ROOT}/Vocabulary/ORBvoc.txt}"
SETTINGS="${SETTINGS:-${SCRIPT_DIR}/q3hy_stereo+imu.yaml}"
BIN="${BIN:-${SCRIPT_DIR}/stereo_inertial_xreal_glass_tag}"

DEFAULT_SEQ="/home/geneta/dataset/01-1/raw_data/glass"
SEQ_PATH="${1:-${SEQ:-${DEFAULT_SEQ}}}"
TRAJ_NAME="${2:-glass_stereo_inertial}"

if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: binary not found: ${BIN}"
  echo "Build first, e.g.:"
  echo "  cmake --build ${ORB_ROOT}/build --target stereo_inertial_xreal_glass_tag -j"
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

echo "VOC      = ${VOC}"
echo "SETTINGS = ${SETTINGS}"
echo "SEQ      = ${SEQ_PATH}"
echo "TRAJ     = ${TRAJ_NAME}"
echo "BIN      = ${BIN}"
echo "-------"

cd "${SCRIPT_DIR}"
exec "${BIN}" "${VOC}" "${SETTINGS}" "${SEQ_PATH}" "${TRAJ_NAME}"
