#!/bin/bash
set -eo pipefail

echo "=== veRL GRPO Training (Megatron Backend) ==="

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1

# --- Configuration ---
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-0.6B}
MODEL_NAME=$(basename "$MODEL_ID")
TOTAL_STEPS=${TOTAL_STEPS:-3}
NNODES=${NNODES:-1}
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}

# Auto-detect NPU count
DEVICE_COUNT=$(python3 -c "import torch_npu; print(torch_npu.npu.device_count())")
echo "NPU devices: $DEVICE_COUNT, Nodes: $NNODES, TP: $TP_SIZE, PP: $PP_SIZE"

# Find model path
MODEL_PATH=${MODEL_PATH:-""}
if [ -z "$MODEL_PATH" ]; then
    for path in /home/weights/$MODEL_NAME $HOME/models/$MODEL_NAME; do
        [ -f "$path/config.json" ] && MODEL_PATH="$path" && break
    done
fi
[ -z "$MODEL_PATH" ] && echo "[FAIL] Model $MODEL_NAME not found. Run prepare-data.sh first." && exit 1

# Find mcore checkpoint
MCORE_DIR=${MCORE_DIR:-$HOME/mcore_ckpt/$MODEL_NAME}
[ ! -d "$MCORE_DIR" ] && echo "[FAIL] Mcore checkpoint not found at $MCORE_DIR. Run prepare-data.sh first." && exit 1

# Find data
DATA_DIR=${DATA_DIR:-$HOME/data/gsm8k}
[ ! -f "$DATA_DIR/train.parquet" ] && echo "[FAIL] Data not found at $DATA_DIR. Run prepare-data.sh first." && exit 1

# Find verl
VERL_DIR=""
for candidate in /verl $HOME/verl /home/*/verl $(pwd)/verl; do
    [ -f "$candidate/verl/__init__.py" ] && VERL_DIR="$candidate" && break
done
[ -z "$VERL_DIR" ] && echo "[FAIL] verl not found." && exit 1

CKPT_DIR=${CKPT_DIR:-$HOME/verl_checkpoints}

cd "$VERL_DIR"
echo "Model: $MODEL_PATH | Mcore: $MCORE_DIR | Data: $DATA_DIR | Steps: $TOTAL_STEPS"

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_megatron_trainer.yaml \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$MCORE_DIR \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.grad_offload=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.strategy=megatron \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$MCORE_DIR \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=verl_grpo \
    trainer.experiment_name=${MODEL_NAME}_grpo_megatron_npu \
    trainer.n_gpus_per_node=$DEVICE_COUNT \
    trainer.nnodes=$NNODES \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.default_local_dir=$CKPT_DIR

echo "=== Training complete ==="
echo "Checkpoints at: $CKPT_DIR/"
