#!/usr/bin/env bash
# RSNA knee competition data (Kaggle): accept the competition rules on the website first, put your API token in
# ~/.kaggle/kaggle.json, then:  bash scripts/01_download_rsna.sh   -> DATA/raw/rsna (train.csv, train_series.csv, train_series/, ...)
set -euo pipefail
DATA="${RSNA_KNEE_DATA:-$(cd "$(dirname "$0")/.." && pwd)/data}"
mkdir -p "$DATA/raw/rsna"
kaggle competitions download -c rsna-knee-abnormality-detection -p "$DATA/raw/rsna"
cd "$DATA/raw/rsna" && unzip -q -n rsna-knee-abnormality-detection.zip && echo "RSNA data in $DATA/raw/rsna"
