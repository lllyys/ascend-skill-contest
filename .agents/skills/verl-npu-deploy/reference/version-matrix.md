# Version Matrix (Tested on Atlas 910B3)

| Package | Version | Source |
|---------|---------|--------|
| CANN | 8.5.0 | Pre-installed in base image |
| Python | 3.11 | Pre-installed in base image |
| torch | 2.8.0 | `pip install torch_npu==2.8.0` (pulls torch) |
| torch_npu | 2.8.0 | `pip install torch_npu==2.8.0` |
| torchvision | 0.23.0 | `pip install torchvision==0.23.0` |
| transformers | 4.57.6 | `pip install transformers==4.57.6` |
| triton-ascend | 3.2.0 | `pip install triton-ascend==3.2.0` |
| setuptools | 80.10.2 | `pip install setuptools==80.10.2` |
| vllm | 0.13.0 | Source build: `VLLM_TARGET_DEVICE=empty` |
| vllm-ascend | releases/v0.13.0 | Source build: `COMPILE_CUSTOM_KERNELS=1` |
| Megatron-LM | core_v0.12.1 | `git clone --branch core_v0.12.1` |
| MindSpeed | 2.3.0_core_r0.12.1 | `git checkout 2.3.0_core_r0.12.1` |
| mbridge | latest | `pip install mbridge` |
| verl | latest | Source build, use `use_dist_checkpointing=True` with mcore-converted weights |

## Known Version Conflicts

1. vllm build requirements pull `torch==2.9.0` — must reinstall `torch==2.8.0` after
2. vllm-ascend requirements pull `torch_npu==2.8.0.post2` and `torchvision==0.23.0` — must reinstall correct versions after
3. Megatron backend requires HF→mcore weight conversion (`converter_hf_to_mcore.py`) and `use_dist_checkpointing=True`
4. `setuptools` must be pinned to 80.10.2 — newer versions break some builds
5. `triton` (CUDA version) must be uninstalled before installing `triton-ascend`


## Unsupported on Ascend

- flash_attn (use MindSpeed flash attention via `use_flash_attn=True` override)
- liger-kernel
