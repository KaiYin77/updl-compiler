#!/bin/bash
# Downoad the dataset.
mkdir -p /home/kaiyin-upbeat/data
cd /home/kaiyin-upbeat/data
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz