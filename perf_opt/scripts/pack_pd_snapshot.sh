#!/bin/bash
# One-click snapshot: pack P/D MTP benchmark artifacts
# Run on remote: bash /tmp/pd_test/pack_pd_snapshot.sh
set -e

DIR="/tmp/pd_test"
OUT="/tmp/pd_bench"
STAMP=$(date +%Y%m%d_%H%M)
TAR="pd_mtp_snapshot_${STAMP}.tar.gz"

cd /tmp

echo "=== Packing P/D MTP Snapshot ==="
tar -czf "$TAR" \
  pd_bench/*.json \
  pd_test/*.json \
  pd_test/*.sh \
  pd_test/*.log \
  pd_test/*.py \
  2>/dev/null || true

echo "Snapshot: /tmp/$TAR"
ls -lh "/tmp/$TAR"