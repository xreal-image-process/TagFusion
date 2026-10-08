#!/usr/bin/env bash
# 启动 XREAL Glass cam0+cam1 TagFusion（双目）
#
# 依赖（本机本地前缀）:
#   source /home/jxpang/project/deps/env.sh
#
# 用法:
#   ./run_xreal_glass_stereo.sh
#   ./run_xreal_glass_stereo.sh /path/to/glass [traj_name]
#   ./run_xreal_glass_stereo.sh /path/to/seq_root [traj_name]   # 含 raw_data/glass 的序列根目录也可
#   ./run_xreal_glass_stereo.sh /path/to/glass/cam0/images [traj_name]
#   SEQ=/path/to/.../raw_data/glass ./run_xreal_glass_stereo.sh
#
# 换配置（默认 XREAL_Glass_stereo.yaml）:
#   SETTINGS=./q3hy_20260807_der_pat_led_f_stereo.yaml \
#     ./run_xreal_glass_stereo.sh \
#     /data/NEW_AI_DATA/data_sip/dataset/q3hy/20260807/der_pat_led_f/01-1 \
#     der_pat_01-1
#
# 环境变量（均可选）:
#   VOC / SETTINGS / BIN / SEQ
#
# 运行前请确认所用 yaml 中 Camera1/2、Stereo.T_c1_c2、分辨率与序列一致
# （q3hy 为 640x512@60；旧 room-patrol 样例为 480x640@30，勿混用）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORB_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

VOC="${VOC:-${ORB_ROOT}/Vocabulary/ORBvoc.txt}"
SETTINGS="${SETTINGS:-${SCRIPT_DIR}/XREAL_Glass_stereo.yaml}"
BIN="${BIN:-${SCRIPT_DIR}/stereo_xreal_glass_tag}"

DEFAULT_SEQ="/home/geneta/dataset/20260227-room-patrol-11-1/raw_data/glass"
SEQ_PATH="${1:-${SEQ:-${DEFAULT_SEQ}}}"
TRAJ_NAME="${2:-glass_stereo}"

if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: binary not found: ${BIN}"
  echo "Build first, e.g.:"
  echo "  cmake --build ${ORB_ROOT}/build --target stereo_xreal_glass_tag -j"
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

if grep -qE 'Camera1\.fx:\s*0(\.0)?\s*$' "${SETTINGS}"; then
  echo "WARNING: ${SETTINGS} 中 Camera1.fx 仍为 0，请先填写鱼眼内参/畸变再跑正式实验。"
fi

echo "VOC      = ${VOC}"
echo "SETTINGS = ${SETTINGS}"
echo "SEQ      = ${SEQ_PATH}"
echo "TRAJ     = ${TRAJ_NAME}"
echo "BIN      = ${BIN}"
echo "-------"

cd "${SCRIPT_DIR}"
exec "${BIN}" "${VOC}" "${SETTINGS}" "${SEQ_PATH}" "${TRAJ_NAME}"
