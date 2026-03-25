#!/bin/bash
set -eo pipefail

echo "=== veRL NPU Install ==="
START_TIME=$(date +%s)

# Source CANN env
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

WORKDIR=${WORKDIR:-$(pwd)}

# Step 1: System deps + Python packaging (combined)
echo "--- Step 1: System deps ---"
apt-get update -qq && apt-get install -y -qq gcc g++ cmake libnuma-dev wget git curl build-essential > /dev/null 2>&1 || true
pip install -q --upgrade pip packaging "setuptools==80.10.2"
echo "[OK] System deps ($(($(date +%s) - START_TIME))s)"

# Step 2: torch_npu (installs torch==2.8.0 automatically)
echo "--- Step 2: torch_npu ---"
if python3 -c "import torch_npu; assert torch_npu.__version__.startswith('2.8')" 2>/dev/null; then
    echo "[SKIP] torch_npu 2.8.x already installed"
else
    pip install -q torch_npu==2.8.0 torchvision==0.23.0 transformers==4.57.6
    echo "[OK] torch_npu ($(($(date +%s) - START_TIME))s)"
fi

# Step 3: Clone all repos in parallel (if not present)
echo "--- Step 3: Clone repos ---"
cd "$WORKDIR"
NEED_CLONE=false
[ -d vllm ] || NEED_CLONE=true
[ -d vllm-ascend ] || NEED_CLONE=true
[ -d MindSpeed ] || NEED_CLONE=true
[ -d Megatron-LM ] || NEED_CLONE=true
[ -d verl ] || NEED_CLONE=true

if [ "$NEED_CLONE" = "true" ]; then
    [ -d vllm ] || git clone --depth 1 --branch v0.13.0 https://github.com/vllm-project/vllm.git &
    [ -d vllm-ascend ] || git clone --depth 1 -b releases/v0.13.0 https://github.com/vllm-project/vllm-ascend.git &
    [ -d MindSpeed ] || git clone --depth 1 -b 2.3.0_core_r0.12.1 https://gitcode.com/Ascend/MindSpeed.git &
    [ -d Megatron-LM ] || git clone --depth 1 --branch core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git &
    [ -d verl ] || git clone --recursive --depth 1 https://github.com/verl-project/verl.git &
    echo "Cloning repos in parallel..."
    wait
    echo "[OK] All repos cloned ($(($(date +%s) - START_TIME))s)"
else
    echo "[SKIP] All repos already present"
fi

# Step 4: vllm
echo "--- Step 4: vllm ---"
if python3 -c "import vllm" 2>/dev/null; then
    echo "[SKIP] vllm already installed"
else
    cd "$WORKDIR/vllm"
    pip install -q -r requirements/build.txt
    VLLM_TARGET_DEVICE=empty pip install -e . 2>&1 | tail -1
    cd "$WORKDIR"
    echo "[OK] vllm ($(($(date +%s) - START_TIME))s)"
fi

# Step 5: vllm-ascend
echo "--- Step 5: vllm-ascend ---"
if python3 -c "from vllm_ascend import _build_info" 2>/dev/null; then
    echo "[SKIP] vllm-ascend already installed"
else
    cd "$WORKDIR/vllm-ascend"
    pip install -q -r requirements.txt
    SOC_VERSION=$(npu-smi info 2>/dev/null | grep -i 'Name' | head -1 | awk '{print tolower($NF)}')
    export SOC_VERSION=${SOC_VERSION:-ascend910b3}
    COMPILE_CUSTOM_KERNELS=1 pip install -e . 2>&1 | tail -1
    cd "$WORKDIR"
    echo "[OK] vllm-ascend ($(($(date +%s) - START_TIME))s)"
fi

# Step 6: MindSpeed + Megatron + triton + mbridge + verl (batch install)
echo "--- Step 6: MindSpeed + Megatron + verl ---"
cd "$WORKDIR"
python3 -c "import mindspeed" 2>/dev/null || pip install -q -e MindSpeed -e Megatron-LM
pip uninstall -y triton 2>/dev/null || true
pip install -q triton-ascend==3.2.0 mbridge 2>/dev/null || true

# verl: install or update
VERL_NEEDS_UPDATE=false
if python3 -c "import verl" 2>/dev/null; then
    HAS_FIX=$(python3 -c "
import inspect
try:
    import verl.utils.megatron.router_replay_utils as rr
    src = inspect.getsource(rr.get_moe_num_layers_to_build)
    print('yes' if 'inspect.signature' in src else 'no')
except: print('no')
" 2>/dev/null)
    [ "$HAS_FIX" = "yes" ] && echo "[SKIP] verl OK" || VERL_NEEDS_UPDATE=true
else
    VERL_NEEDS_UPDATE=true
fi

if [ "$VERL_NEEDS_UPDATE" = "true" ]; then
    cd "$WORKDIR/verl"
    [ -d .git ] && git pull origin main 2>&1 | tail -1 || true
    pip install -q -r requirements-npu.txt && pip install -e . 2>&1 | tail -1
    cd "$WORKDIR"
fi
echo "[OK] All deps ($(($(date +%s) - START_TIME))s)"

# Final verification
echo "--- Verify ---"
python3 -c "
import torch; print('torch:', torch.__version__)
import torch_npu; print('torch_npu:', torch_npu.__version__)
import vllm; print('vllm:', getattr(vllm, '__version__', 'OK'))
from vllm_ascend import _build_info; print('vllm_ascend: OK')
import mindspeed; print('mindspeed: OK')
import verl; print('verl:', getattr(verl, '__version__', 'OK'))
"
echo "=== Install complete in $(($(date +%s) - START_TIME))s ==="
