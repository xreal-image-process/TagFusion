#!/usr/bin/env bash
# Run Tank short_test stereo TagFusion.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
VOC="${ROOT}/Vocabulary/ORBvoc.txt"
SETTINGS="$(cd "$(dirname "$0")" && pwd)/test.yaml"
# Prefer extracted Tank sequence; fall back to local mav0 symlink layout
SEQ="${1:-/home/geneta/下载/Tank/short_test}"
TRAJ_NAME="${2:-short_test_stereo}"

BIN="$(cd "$(dirname "$0")" && pwd)/stereo_test_tag"
if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: ${BIN} not found. Build with: cmake --build build --target stereo_test_tag"
  exit 1
fi

cd "$(dirname "$0")"
echo "VOC     = ${VOC}"
echo "SETTINGS= ${SETTINGS}"
echo "SEQ     = ${SEQ}"
echo "TRAJ    = ${TRAJ_NAME}"
echo "BIN     = ${BIN}"
exec "${BIN}" "${VOC}" "${SETTINGS}" "${SEQ}" "${TRAJ_NAME}"
