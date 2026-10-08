#!/usr/bin/env bash
# Run PennCOSYVIO VI-Sensor stereo TagFusion (default: visensor/af).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
VOC="${ROOT}/Vocabulary/ORBvoc.txt"
SETTINGS="$(cd "$(dirname "$0")" && pwd)/PennCOSYVIO_VI_Stereo.yaml"
SEQ="${1:-/home/geneta/下载/PennCOSYVIO/visensor/af}"
TRAJ_NAME="${2:-af_stereo}"

BIN="$(cd "$(dirname "$0")" && pwd)/stereo_penncosyvio_tag"
if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: ${BIN} not found. Build with: cmake --build build --target stereo_penncosyvio_tag"
  exit 1
fi

cd "$(dirname "$0")"
echo "VOC     = ${VOC}"
echo "SETTINGS= ${SETTINGS}"
echo "SEQ     = ${SEQ}"
echo "TRAJ    = ${TRAJ_NAME}"
echo "BIN     = ${BIN}"
exec "${BIN}" "${VOC}" "${SETTINGS}" "${SEQ}" "${TRAJ_NAME}"
