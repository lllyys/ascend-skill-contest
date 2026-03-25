#!/bin/bash
set -eo pipefail

echo "=== Prepare Data & Model ==="

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-0.6B}
MODEL_NAME=$(basename "$MODEL_ID")
DATA_DIR=${DATA_DIR:-$HOME/data/gsm8k}
MODEL_DIR=${MODEL_PATH:-""}
MCORE_DIR=${MCORE_DIR:-$HOME/mcore_ckpt/$MODEL_NAME}

# Find verl directory
VERL_DIR=""
for candidate in /verl $HOME/verl /home/*/verl $(pwd)/verl; do
    [ -f "$candidate/verl/__init__.py" ] && VERL_DIR="$candidate" && break
done
[ -z "$VERL_DIR" ] && echo "[FAIL] verl directory not found" && exit 1

# Find model (check common locations)
if [ -z "$MODEL_DIR" ]; then
    for path in /home/weights/$MODEL_NAME $HOME/models/$MODEL_NAME $HOME/.cache/huggingface/hub/models--${MODEL_ID//\//-}; do
        [ -f "$path/config.json" ] && MODEL_DIR="$path" && break
    done
fi

# 1. Prepare GSM8K data
if [ -f "$DATA_DIR/train.parquet" ] && [ -f "$DATA_DIR/test.parquet" ]; then
    echo "[OK] GSM8K data at $DATA_DIR"
else
    echo "Preparing GSM8K data..."
    cd "$VERL_DIR"
    python3 examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR"
    echo "[OK] GSM8K data saved to $DATA_DIR"
fi

# 2. Download model if not found
if [ -n "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
    echo "[OK] Model found at $MODEL_DIR"
else
    echo "Downloading $MODEL_ID..."
    MODEL_DIR=$HOME/models/$MODEL_NAME
    pip install -q huggingface_hub 2>/dev/null || true
    huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR"
    echo "[OK] Model downloaded to $MODEL_DIR"
fi

# 3. Convert HF weights to Megatron-Core format
if [ -d "$MCORE_DIR" ] && [ "$(ls -A "$MCORE_DIR" 2>/dev/null)" ]; then
    echo "[OK] Megatron-Core checkpoint at $MCORE_DIR"
else
    echo "Converting $MODEL_NAME to Megatron-Core format..."
    cd "$VERL_DIR"
    python3 scripts/converter_hf_to_mcore.py \
        --hf_model_path "$MODEL_DIR" \
        --output_path "$MCORE_DIR"
    echo "[OK] Converted to $MCORE_DIR"
fi

echo ""
echo "DATA_DIR=$DATA_DIR"
echo "MODEL_DIR=$MODEL_DIR"
echo "MCORE_DIR=$MCORE_DIR"
echo "=== Data preparation complete ==="
