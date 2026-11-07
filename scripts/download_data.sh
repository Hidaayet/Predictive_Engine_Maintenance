#!/usr/bin/env bash
# Downloads the NASA C-MAPSS files (public domain) from a GitHub mirror of the original NASA release.
# Original source: NASA Prognostics Data Repository, "Turbofan Engine Degradation Simulation Data Set" (C-MAPSS).
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p data/raw; cd data/raw
BASE="https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master"
for f in train test RUL; do for n in 1 2 3 4; do
  [ -s "${f}_FD00${n}.txt" ] || curl -fsSL -O "$BASE/${f}_FD00${n}.txt"
done; done
# sanity: published row counts of the train files
test "$(wc -l < train_FD001.txt)" = 20631 && test "$(wc -l < train_FD002.txt)" = 53759 && test "$(wc -l < train_FD004.txt)" = 61249
echo "data ok"
