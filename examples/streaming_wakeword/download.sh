#!/bin/bash
# Download datasets for streaming wake word detection training

DATA_DIR="/home/kaiyin-upbeat/data"
echo "Setting up datasets in $DATA_DIR"
mkdir -p $DATA_DIR
cd $DATA_DIR

# Download Speech Commands v2 dataset
echo "=== Downloading Speech Commands v2 dataset ==="
if [ ! -f "speech_commands_v0.02.tar.gz" ]; then
    wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
else
    echo "Speech Commands dataset already downloaded"
fi

# Extract Speech Commands dataset
if [ ! -d "speech_commands" ]; then
    echo "Extracting Speech Commands dataset..."
    mkdir -p speech_commands
    cd speech_commands
    tar -xzf ../speech_commands_v0.02.tar.gz
    cd ..
    echo "Speech Commands extracted to $DATA_DIR/speech_commands"
else
    echo "Speech Commands already extracted"
fi

# Download MUSAN dataset for background noise augmentation
echo "=== Downloading MUSAN dataset for background noise ==="
if [ ! -f "musan.tar.gz" ]; then
    echo "Downloading MUSAN dataset (approximately 12GB)..."
    wget https://www.openslr.org/resources/17/musan.tar.gz
else
    echo "MUSAN dataset already downloaded"
fi

# Extract MUSAN dataset
if [ ! -d "musan" ]; then
    echo "Extracting MUSAN dataset..."
    tar -xzf musan.tar.gz
    echo "MUSAN extracted to $DATA_DIR/musan"
else
    echo "MUSAN already extracted"
fi

# Verify the expected directory structure
echo "=== Verifying dataset structure ==="
echo "Speech Commands structure:"
ls -la speech_commands/ | head -10

echo ""
echo "MUSAN structure:"
if [ -d "musan" ]; then
    ls -la musan/
    echo ""
    echo "MUSAN noise subdirectories:"
    ls -la musan/noise/ 2>/dev/null || echo "noise/ not found"
    echo ""
    echo "MUSAN speech subdirectories:"
    ls -la musan/speech/ 2>/dev/null || echo "speech/ not found"
fi

echo ""
echo "=== Dataset Setup Complete ==="
echo "Speech Commands: $DATA_DIR/speech_commands"
echo "MUSAN (background noise): $DATA_DIR/musan"
echo ""
echo "Note: MUSAN dataset is large (~12GB). Background noise augmentation"
echo "will use subdirectories: noise/free-sound, speech/us-gov, etc."