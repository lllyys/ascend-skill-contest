---
name: verl-npu-deploy
description: Deploys the veRL RLHF framework on Huawei Ascend NPU with Megatron+MindSpeed training backend and vLLM+vllm-ascend rollout backend. Guides environment setup, dependency installation, weight conversion, GRPO training with checkpoint saving, and checkpoint resume verification. Use when the user needs to deploy veRL on Ascend NPU, run RLHF/GRPO training, or verify checkpoint save/load workflows. Triggers on: Verl 环境部署 → 训练启动 → checkpoint 验证, veRL/Verl deployment, 昇腾 NPU 环境部署, 断点续训, GRPO, RLHF training on Ascend.
---

# veRL NPU Deployment

Deploy veRL on Ascend NPU with Megatron+MindSpeed (training) and vLLM+vllm-ascend (rollout).

## Prerequisites

Ask the user for:
1. **Model** (default: `Qwen/Qwen3-0.6B`)
2. **Model path** (auto-search `/home/weights/`, `~/models/`)
3. **Container** — existing or build from CANN base?

If not specified, state defaults and wait for confirmation. NPU count auto-detected.

## Workflow

1. Source CANN env + preflight checks
2. Install dependencies (**use parallel tracks to save time**)
3. Prepare data + convert weights
4. Train + verify checkpoint save + resume
5. Report results

## Step 1: Environment

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

Check NPU: `npu-smi info -l` (need ≥ 2 devices)
Check shm: `df -h /dev/shm` (need ≥ 16G, else recreate container with `--shm-size=32g`)

## Step 2: Install Dependencies

Skip any package already installed. **IMPORTANT: Use parallel installation to save time.**

### 2.1 Base (sequential — must complete first)
```bash
apt-get update && apt-get install -y gcc g++ cmake libnuma-dev wget git curl build-essential
pip install --upgrade pip packaging "setuptools==80.10.2"
pip install torch_npu==2.8.0 torchvision==0.23.0 transformers==4.57.6
```
DO NOT install torch separately — `torch_npu==2.8.0` pulls the correct `torch==2.8.0`.

### 2.2 Clone all repos in parallel
```bash
git clone --depth 1 --branch v0.13.0 https://github.com/vllm-project/vllm.git &
git clone --depth 1 -b releases/v0.13.0 https://github.com/vllm-project/vllm-ascend.git &
git clone --depth 1 -b 2.3.0_core_r0.12.1 https://gitcode.com/Ascend/MindSpeed.git &
git clone --depth 1 --branch core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git &
git clone --recursive --depth 1 https://github.com/verl-project/verl.git &
wait
```

### 2.3 Parallel install tracks

After clones complete, run these THREE tracks in parallel:

**Track A** (slow, ~15 min — vllm + vllm-ascend):
```bash
cd vllm && pip install -r requirements/build.txt && VLLM_TARGET_DEVICE=empty pip install -e . && cd ..
cd vllm-ascend && pip install -r requirements.txt && SOC_VERSION=${SOC_VERSION:-ascend910b3} COMPILE_CUSTOM_KERNELS=1 pip install -e . && cd ..
```

**Track B** (fast, ~3 min — MindSpeed + Megatron + triton + mbridge + verl):
```bash
pip install -e MindSpeed -e Megatron-LM
pip uninstall -y triton 2>/dev/null; pip install triton-ascend==3.2.0 mbridge
cd verl && pip install -r requirements-npu.txt && pip install -e . && cd ..
```

**Track C** (fast, ~2 min — data + weight conversion, after verl from Track B):
```bash
cd verl && python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k && cd ..
python3 verl/scripts/converter_hf_to_mcore.py --hf_model_path $MODEL_PATH --output_path ~/mcore_ckpt/$(basename $MODEL_PATH)
```

**Track A and Track B can run simultaneously.** Track C can start as soon as Track B's verl install finishes.

### Verify
```bash
python3 -c "import verl, vllm, mindspeed, torch_npu; print('OK')"
```

## Step 3: Train

### Environment variables
```bash
export VLLM_USE_V1=1 TASK_QUEUE_ENABLE=1 CPU_AFFINITY_CONF=1
```
Do NOT set `VLLM_ATTENTION_BACKEND`.

### Auto-detect NPU count
```bash
DEVICE_COUNT=$(python3 -c "import torch_npu; print(torch_npu.npu.device_count())")
```

For models ≤7B: TP=1, PP=1. Larger: increase accordingly.

### Launch training
```bash
cd verl && python3 -m verl.trainer.main_ppo \
    --config-path=config --config-name=ppo_megatron_trainer.yaml \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=16 data.max_prompt_length=512 data.max_response_length=128 \
    data.filter_overlong_prompts=True data.truncation=error \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$MCORE_CKPT \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.grad_offload=False \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.strategy=megatron \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$MCORE_CKPT \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    algorithm.kl_ctrl.kl_coef=0.001 trainer.critic_warmup=0 trainer.logger=console \
    trainer.project_name=verl_grpo \
    trainer.experiment_name=$(basename $MODEL_PATH)_grpo_megatron_npu \
    trainer.n_gpus_per_node=$DEVICE_COUNT trainer.nnodes=${NNODES:-1} \
    trainer.save_freq=1 trainer.test_freq=-1 \
    trainer.total_epochs=1 trainer.total_training_steps=3 \
    trainer.default_local_dir=$HOME/verl_checkpoints
```

## Step 4: Verify Checkpoint

Check saved:
```bash
ls ~/verl_checkpoints/ && cat ~/verl_checkpoints/latest_checkpointed_iteration.txt
```

Resume: run the same training command with `total_training_steps=6`. Verify new checkpoint step > previous.

## Report Results

After completion, tell the user:
- Environment: CANN version, NPU count, container
- Model: HF path, mcore checkpoint path
- Training: steps completed, algorithm
- Checkpoints: directory, latest step
- Resume: previous step → new step, PASSED/FAILED
- List all output file paths

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Engine core init failed | Kill stale NPU processes; `/dev/shm` ≥ 16G; `TASK_QUEUE_ENABLE=1` |
| Not enough memory for cache | `gpu_memory_utilization=0.9` |
| Undefined soc_version | Lowercase: `ascend910b3` not `Ascend910B3` |
| Cannot import `_build_info` | Rebuild: `SOC_VERSION=ascend910b3 COMPILE_CUSTOM_KERNELS=1 pip install -e .` |
| Invalid XFORMERS backend | `unset VLLM_ATTENTION_BACKEND` |
| Weights not found in safetensors | Convert HF→mcore first: `python3 scripts/converter_hf_to_mcore.py` |
| vp_stage TypeError | Update verl: `cd verl && git pull && pip install -e .` |
| Bridge.save_weights error | Use `use_dist_checkpointing=True` |
| FRACTAL_NZ enabled | `export VLLM_ASCEND_ENABLE_NZ=0` |
| Model too large for single NPU | Increase TP/PP |
| Network unreachable | Set proxy: `export http_proxy=http://127.0.0.1:58231 https_proxy=http://127.0.0.1:58231` |

## Docker

```bash
docker run -dit --name <name> --privileged --shm-size=32g \
    -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    --network host <cann-base-image> bash
```

## References

- [reference/version-matrix.md](reference/version-matrix.md)
- [reference/troubleshooting.md](reference/troubleshooting.md)
