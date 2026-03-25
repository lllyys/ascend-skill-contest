#!/bin/bash
set -e

echo "=== veRL NPU Preflight Check ==="

# 1. Source CANN env
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    echo "[OK] CANN ascend-toolkit env sourced"
else
    echo "[FAIL] CANN ascend-toolkit not found at /usr/local/Ascend/ascend-toolkit/set_env.sh"
    exit 1
fi

if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
    echo "[OK] CANN nnal/atb env sourced"
else
    echo "[WARN] nnal/atb env not found — some features may not work"
fi

# 2. Check NPU devices (use npu-smi if torch_npu not yet installed)
if python3 -c "import torch_npu; print(torch_npu.npu.device_count())" 2>/dev/null | grep -qE '^[2-9]|^[0-9]{2,}'; then
    DEVICE_COUNT=$(python3 -c "import torch_npu; print(torch_npu.npu.device_count())")
    echo "[OK] NPU devices: $DEVICE_COUNT"
elif npu-smi info -l 2>/dev/null | grep -q "Total Count"; then
    DEVICE_COUNT=$(npu-smi info -l 2>/dev/null | grep "Total Count" | head -1 | awk '{print $NF}')
    echo "[OK] NPU devices (via npu-smi): $DEVICE_COUNT"
    if [ "${DEVICE_COUNT:-0}" -lt 2 ]; then
        echo "[FAIL] Need >= 2 NPU devices, found: $DEVICE_COUNT"
        exit 1
    fi
else
    echo "[WARN] Cannot detect NPU count — torch_npu not installed yet and npu-smi not available"
    echo "       Will verify after install"
fi

# 3. Check shared memory
SHM_SIZE=$(df -BG /dev/shm 2>/dev/null | tail -1 | awk '{print $2}' | tr -d 'G')
if [ "${SHM_SIZE:-0}" -ge 16 ]; then
    echo "[OK] /dev/shm: ${SHM_SIZE}G"
else
    echo "[FAIL] /dev/shm too small: ${SHM_SIZE:-unknown}G (need >= 16G)"
    echo "       Recreate container with --shm-size=32g"
    exit 1
fi

# 4. Check stale NPU processes
STALE=$(npu-smi info 2>/dev/null | grep -c "Process id" || true)
if [ "$STALE" -gt 1 ]; then
    echo "[WARN] Stale NPU processes detected — may cause OOM. Run 'npu-smi info' to check."
fi

# 5. Check core deps (informational, not blocking)
python3 -c "import torch; print('[OK] torch:', torch.__version__)" 2>/dev/null || echo "[INFO] torch not installed — run install.sh"
python3 -c "import torch_npu; print('[OK] torch_npu:', torch_npu.__version__)" 2>/dev/null || echo "[INFO] torch_npu not installed — run install.sh"
python3 -c "import verl; print('[OK] verl installed')" 2>/dev/null || echo "[INFO] verl not installed — run install.sh"
python3 -c "import vllm; print('[OK] vllm installed')" 2>/dev/null || echo "[INFO] vllm not installed — run install.sh"
python3 -c "import mindspeed; print('[OK] mindspeed installed')" 2>/dev/null || echo "[INFO] mindspeed not installed — run install.sh"

echo "=== Preflight complete ==="
