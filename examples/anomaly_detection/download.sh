#!/bin/bash
set -euo pipefail

DATA_ROOT="/home/kaiyin-upbeat/data/anomaly_detection"
DEV_ZIP="dev_data_ToyCar.zip"
DEV_URL="https://zenodo.org/record/3678171/files/${DEV_ZIP}?download=1"
EVAL_ZIP="eval_data_train_ToyCar.zip"
EVAL_URL="https://zenodo.org/record/3727685/files/${EVAL_ZIP}?download=1"

# Install unzip if not present
if ! command -v unzip >/dev/null 2>&1; then
  echo "Installing unzip..."
  sudo apt-get update && sudo apt-get install -y unzip
fi

mkdir -p "${DATA_ROOT}"
cd "${DATA_ROOT}"

echo "Downloading ToyCar development data..."
curl -L "${DEV_URL}" -o "${DEV_ZIP}" || wget "${DEV_URL}" -O "${DEV_ZIP}"
unzip -o "${DEV_ZIP}" -d "dev_data"
rm -f "${DEV_ZIP}"

echo "Downloading ToyCar evaluation training data..."
curl -L "${EVAL_URL}" -o "${EVAL_ZIP}" || wget "${EVAL_URL}" -O "${EVAL_ZIP}"
unzip -o "${EVAL_ZIP}" -d "dev_data"
rm -f "${EVAL_ZIP}"

echo "Dataset ready under ${DATA_ROOT}/dev_data"