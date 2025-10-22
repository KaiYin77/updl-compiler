#!/bin/bash
# Downoad the dataset.
mkdir -p /home/kaiyin-upbeat/data
cd /home/kaiyin-upbeat/data
wget https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz
tar -xvf vw_coco2014_96.tar.gz