#!/usr/bin/env bash
# Run from the repo root (the folder containing srm_project/). Moves legacy code into srm/legacy/.
set -euo pipefail
mkdir -p srm/legacy
for f in dataset models_v3 segmentation evaluation_v2 main fetch_real_tile; do
  git mv "srm_project/$f.py" "srm/legacy/$f.py"
done
git mv srm_project/README.md srm/legacy/README_legacy_verification_log.md
git mv srm_project/requirements.txt srm/legacy/requirements_legacy.txt
