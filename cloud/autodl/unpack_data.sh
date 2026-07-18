#!/usr/bin/env bash
set -euo pipefail

ARCHIVE="${1:-}"
DATA_ROOT="${2:-/root/autodl-tmp/shd-af-clean-data}"

if [[ -z "$ARCHIVE" ]]; then
  echo "Usage: bash cloud/autodl/unpack_data.sh /path/to/shd-af-clean-data.tar.gz [data_root]"
  exit 2
fi

mkdir -p "$(dirname "$DATA_ROOT")"

case "$ARCHIVE" in
  *.tar.gz|*.tgz)
    tar -xzf "$ARCHIVE" -C "$(dirname "$DATA_ROOT")"
    ;;
  *.tar)
    tar -xf "$ARCHIVE" -C "$(dirname "$DATA_ROOT")"
    ;;
  *.zip)
    python - "$ARCHIVE" "$(dirname "$DATA_ROOT")" <<'PY'
import sys
from zipfile import ZipFile

archive, out_dir = sys.argv[1], sys.argv[2]
with ZipFile(archive) as zf:
    zf.extractall(out_dir)
PY
    ;;
  *)
    echo "Unsupported archive: $ARCHIVE"
    exit 2
    ;;
esac

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Expected data root not found after unpacking: $DATA_ROOT"
  echo "Check archive top-level directory name."
  exit 1
fi

echo "Unpacked data root: $DATA_ROOT"

