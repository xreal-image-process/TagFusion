#!/usr/bin/env bash
# 用 evo 对 XREAL_Glass 目录下的估计轨迹做 APE 评估（真值：cam0_pose.txt）
#
# 数据集根目录约定：
#   /data/NEW_AI_DATA/data_sip/dataset/q3hy/<date>/<dataset_name>/<seq>/ground_truth/data/cam0_pose.txt
#
# 用法:
#   ./scripts/eval_evo_ape.sh der_pat_led_f 01-3
#   ./scripts/eval_evo_ape.sh der_pat_led_f 01-3 OnlineCameraTrajectory.txt
#   ./scripts/eval_evo_ape.sh der_pat_led_f 01-3 f_der_pat_01-3.txt
#
# 可选环境变量:
#   Q3HY_ROOT   默认 /data/NEW_AI_DATA/data_sip/dataset/q3hy
#   TRAJ_DIR    估计轨迹所在目录，默认 Examples/TagFusion/XREAL_Glass
#   EVO_ARGS    额外传给 evo_ape 的参数，默认 "-a --plot --save_plot"
#               双目米制默认不做尺度对齐；若需尺度对齐: EVO_ARGS="-as --plot"
#   T_MAX_DIFF  时间关联阈值（秒），默认 0.02
#
# 依赖: pip install evo   （需本机可 import evo / 有 evo_ape）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAJ_DIR="${TRAJ_DIR:-${ORB_ROOT}/Examples/TagFusion/XREAL_Glass}"

Q3HY_ROOT="${Q3HY_ROOT:-/data/NEW_AI_DATA/data_sip/dataset/q3hy}"
T_MAX_DIFF="${T_MAX_DIFF:-0.02}"
EVO_ARGS="${EVO_ARGS:--a --plot --save_plot}"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") <dataset_name> <seq> [traj_file]

Examples:
  $(basename "$0") der_pat_led_f 01-3
  $(basename "$0") der_pat_led_f 01-3 OnlineCameraTrajectory.txt
  $(basename "$0") der_pat_led_f 01-3 f_der_pat_01-3.txt

TRAJ_DIR (default):
  ${ORB_ROOT}/Examples/TagFusion/XREAL_Glass

Looks up GT:
  \$Q3HY_ROOT/<date>/<dataset_name>/<seq>/ground_truth/data/cam0_pose.txt
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

DATASET_NAME="$1"
SEQ="$2"
TRAJ_IN="${3:-}"

# --- locate evo ---
if command -v evo_ape >/dev/null 2>&1; then
  EVO_APE=(evo_ape)
elif python3 -c 'import evo' >/dev/null 2>&1; then
  EVO_APE=(python3 -m evo.main_ape)
else
  echo "ERROR: evo 未安装。请先执行:  pip3 install --user evo"
  echo "然后确保 ~/.local/bin 在 PATH 中，或使用: python3 -m evo.main_ape"
  exit 1
fi

if [[ ! -d "${TRAJ_DIR}" ]]; then
  echo "ERROR: TRAJ_DIR 不存在: ${TRAJ_DIR}"
  exit 1
fi

# --- resolve dataset root by short name (prefer newest date that contains seq) ---
resolve_dataset_root() {
  local name="$1" seq="$2"
  local candidates=()
  local d
  for d in "${Q3HY_ROOT}"/*/"${name}"; do
    [[ -d "${d}/${seq}/ground_truth/data" ]] || continue
    candidates+=("$d")
  done
  if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "ERROR: 未找到数据集 ${name}/${seq} under ${Q3HY_ROOT}" >&2
    echo "期望: ${Q3HY_ROOT}/<date>/${name}/${seq}/ground_truth/data/cam0_pose.txt" >&2
    return 1
  fi
  # 按路径名排序，取最后一个（通常日期更大）
  printf '%s\n' "${candidates[@]}" | sort | tail -1
}

DATASET_ROOT="$(resolve_dataset_root "${DATASET_NAME}" "${SEQ}")"
GT_RAW="${DATASET_ROOT}/${SEQ}/ground_truth/data/cam0_pose.txt"
if [[ ! -f "${GT_RAW}" ]]; then
  echo "ERROR: GT 不存在: ${GT_RAW}"
  exit 1
fi

# --- resolve estimated trajectory under TRAJ_DIR ---
pick_traj() {
  local seq="$1" explicit="${2:-}"
  local f
  if [[ -n "${explicit}" ]]; then
    if [[ -f "${explicit}" ]]; then
      echo "$(cd "$(dirname "${explicit}")" && pwd)/$(basename "${explicit}")"
      return 0
    fi
    if [[ -f "${TRAJ_DIR}/${explicit}" ]]; then
      echo "${TRAJ_DIR}/${explicit}"
      return 0
    fi
    echo "ERROR: 轨迹文件不存在: ${explicit}" >&2
    return 1
  fi

  (
    cd "${TRAJ_DIR}"
    for c in \
      "f_${DATASET_NAME}_${seq}.txt" \
      "f_der_pat_${seq}.txt" \
      "f_${seq}.txt" \
      f_*"${seq}"*.txt \
      "OnlineCameraTrajectory.txt" \
      "CameraTrajectory.txt"
    do
      # shellcheck disable=SC2086
      for f in $c; do
        if [[ -f "$f" ]]; then
          echo "${TRAJ_DIR}/${f}"
          exit 0
        fi
      done
    done
    exit 1
  ) && return 0

  echo "ERROR: ${TRAJ_DIR} 下未找到估计轨迹。请指定第 3 个参数，或先跑完生成 OnlineCameraTrajectory.txt / f_*.txt" >&2
  return 1
}

TRAJ="$(pick_traj "${SEQ}" "${TRAJ_IN}")"

OUT_DIR="${TRAJ_DIR}/evo_results/${DATASET_NAME}_${SEQ}"
mkdir -p "${OUT_DIR}"

# evo 的 tum 解析不吃注释行；准备干净 GT / EST
GT_TUM="${OUT_DIR}/gt_cam0_pose.tum"
EST_TUM="${OUT_DIR}/est.tum"
rg -v '^\s*#|^\s*$' "${GT_RAW}" > "${GT_TUM}"
rg -v '^\s*#|^\s*$' "${TRAJ}" > "${EST_TUM}"

echo "DATASET = ${DATASET_NAME}"
echo "SEQ     = ${SEQ}"
echo "ROOT    = ${DATASET_ROOT}"
echo "GT      = ${GT_RAW}"
echo "EST     = ${TRAJ}"
echo "TRAJDIR = ${TRAJ_DIR}"
echo "OUT     = ${OUT_DIR}"
echo "EVO     = ${EVO_APE[*]} tum ... ${EVO_ARGS}"
echo "-------"

# 简单时间重叠检查（同一时间轴：cam 时钟）
python3 - <<PY
from pathlib import Path
def t_range(p):
    ts=[]
    for line in Path(p).read_text().splitlines():
        parts=line.split()
        if len(parts) >= 8:
            ts.append(float(parts[0]))
    return (ts[0], ts[-1], len(ts)) if ts else (None, None, 0)
g0,g1,ng = t_range("${GT_TUM}")
e0,e1,ne = t_range("${EST_TUM}")
print(f"GT  time: [{g0:.6f}, {g1:.6f}]  n={ng}")
print(f"EST time: [{e0:.6f}, {e1:.6f}]  n={ne}")
lo, hi = max(g0,e0), min(g1,e1)
if lo >= hi:
    print("WARNING: 估计轨迹与 cam0_pose 真值时间无重叠。")
    print("  cam0_pose 通常从较晚时刻开始（本序列约 t>=85s）；若只跑了前几十秒，无法评估。")
    print("  请跑完覆盖真值区间后再评估，或换已覆盖该区间的轨迹文件。")
else:
    print(f"overlap: [{lo:.6f}, {hi:.6f}]")
PY

PLOT_FILE="${OUT_DIR}/ape_plot.pdf"
ZIP_FILE="${OUT_DIR}/ape_result.zip"

# shellcheck disable=SC2086
"${EVO_APE[@]}" tum "${GT_TUM}" "${EST_TUM}" \
  ${EVO_ARGS} \
  --t_max_diff "${T_MAX_DIFF}" \
  --save_plot "${PLOT_FILE}" \
  --save_results "${ZIP_FILE}" \
  | tee "${OUT_DIR}/ape_metrics.txt"

echo "-------"
echo "Saved:"
echo "  ${OUT_DIR}/ape_metrics.txt"
echo "  ${PLOT_FILE}"
echo "  ${ZIP_FILE}"
