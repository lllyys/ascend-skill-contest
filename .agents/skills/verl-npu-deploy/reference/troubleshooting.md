# Troubleshooting verl on Ascend NPU

## Engine Core Initialization Failed

**Symptom:** `RuntimeError: Engine core initialization failed. See root cause above.`

**Possible causes:**

1. **Stale NPU processes** — previous training left processes on devices
   - Check: `npu-smi info` — look for running processes
   - Fix: `kill -9 <pid>` from host (not from inside container)

2. **Insufficient shared memory** — container created without `--shm-size`
   - Check: `df -h /dev/shm` — should be ≥16GB
   - Fix: recreate container with `--shm-size=32g`

3. **TASK_QUEUE_ENABLE=2** — not supported during NPU graph capture
   - Fix: `export TASK_QUEUE_ENABLE=1`

4. **Insufficient GPU memory for cache blocks**
   - Fix: increase `gpu_memory_utilization` to `0.9`

## vllm-ascend Build Failures

**Symptom:** `AssertionError: Undefined soc_version: Ascend910B1`

**Fix:** SOC_VERSION must be lowercase: `ascend910b1`

Valid values: `ascend910b1`, `ascend910b2`, `ascend910b3`, `ascend910b4`, `ascend910_9381` (A3)

**Symptom:** `cannot import name '_build_info' from 'vllm_ascend'`

**Fix:** Custom kernels did not compile. Rebuild:
```bash
cd vllm-ascend
SOC_VERSION=ascend910b1 COMPILE_CUSTOM_KERNELS=1 pip install -v -e .
```

## Version Conflicts

**Symptom:** `torch.__version__` shows 2.9.0 after vllm install

**Fix:**
```bash
pip install --no-deps torch==2.8.0 torch_npu==2.8.0 torchvision==0.23.0
```

## Megatron Backend Issues

**Symptom:** `Weights not found in safetensors files`

**Fix:** Megatron backend requires HF weights converted to mcore format:
```bash
cd verl
python3 scripts/converter_hf_to_mcore.py \
    --hf_model_path <model_path> \
    --output_path <mcore_output_path>
```
Then use `use_dist_checkpointing=True` and `dist_checkpointing_path=<mcore_output_path>`.

**Symptom:** `get_transformer_layer_offset() got an unexpected keyword argument 'vp_stage'`

**Cause:** Pre-built Docker images may contain an older verl snapshot missing the Megatron core_v0.12.1 compatibility fix (PR #5580, merged 2026-03-16).

**Fix:** Update verl to latest:
```bash
cd verl && git pull origin main && pip install -v -e .
```
Or run `install.sh` which auto-detects and updates. Building from scratch always gets the fix.

**Symptom:** `Bridge.save_weights() got an unexpected keyword argument 'distributed_filesystem'`

**Fix:** Use `use_dist_checkpointing=True` instead of mbridge-based checkpoint saving.

**Symptom:** `FRACTAL_NZ mode is enabled`

**Fix:** `export VLLM_ASCEND_ENABLE_NZ=0`

## CANN Environment Not Sourced

**Symptom:** Various import errors or device not found

**Fix:** Always source before any operation:
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

## VLLM_ATTENTION_BACKEND Invalid

**Symptom:** `ValidationError: Invalid value 'XFORMERS' for VLLM_ATTENTION_BACKEND`

**Fix:** Do not set this variable. Unset it:
```bash
unset VLLM_ATTENTION_BACKEND
```
