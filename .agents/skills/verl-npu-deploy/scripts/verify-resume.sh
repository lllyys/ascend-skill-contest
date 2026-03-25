#!/bin/bash
set -e

echo "=== Verify Checkpoint & Resume ==="

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true
export VLLM_USE_V1=1 TASK_QUEUE_ENABLE=1 CPU_AFFINITY_CONF=1

CKPT_DIR=${CKPT_DIR:-$HOME/verl_checkpoints}

# 1. Find checkpoint — use experiment name for deterministic lookup
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-0.6B}
MODEL_NAME=$(basename "$MODEL_ID")
EXP_DIR="$CKPT_DIR/verl_grpo/${MODEL_NAME}_grpo_megatron_npu"

echo "--- Check checkpoint ---"
CKPT_FILE=""
# First try experiment-specific path
if [ -f "$EXP_DIR/latest_checkpointed_iteration.txt" ]; then
    CKPT_FILE="$EXP_DIR/latest_checkpointed_iteration.txt"
else
    # Fallback to searching CKPT_DIR
    CKPT_FILE=$(find "$CKPT_DIR" -name "latest_checkpointed_iteration.txt" -type f 2>/dev/null | head -1)
fi

if [ -z "$CKPT_FILE" ]; then
    echo "[FAIL] No checkpoints found at $CKPT_DIR. Run training first."
    exit 1
fi

CKPT_SUBDIR=$(dirname "$CKPT_FILE")
LATEST=$(cat "$CKPT_FILE")
echo "[OK] Latest checkpoint: step $LATEST at $CKPT_SUBDIR"

# 2. Verify checkpoint has actor data
STEP_DIR="$CKPT_SUBDIR/global_step_$LATEST"
[ -d "$STEP_DIR/actor" ] && echo "[OK] Actor checkpoint exists" || { echo "[FAIL] Actor checkpoint missing"; exit 1; }

# 3. Resume training with more steps
echo ""
echo "--- Resuming training (target: step > $LATEST) ---"

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-0.6B}
MODEL_NAME=$(basename "$MODEL_ID")
if [ -z "$MODEL_PATH" ]; then
    for path in /home/weights/$MODEL_NAME $HOME/models/$MODEL_NAME; do
        [ -f "$path/config.json" ] && MODEL_PATH="$path" && break
    done
fi
MCORE_DIR=${MCORE_DIR:-$HOME/mcore_ckpt/$MODEL_NAME}
DATA_DIR=${DATA_DIR:-$HOME/data/gsm8k}
VERL_DIR=""
for candidate in /verl $HOME/verl /home/*/verl; do
    [ -f "$candidate/verl/__init__.py" ] && VERL_DIR="$candidate" && break
done

[ -z "$MODEL_PATH" ] || [ -z "$VERL_DIR" ] && echo "[FAIL] Cannot find model or verl" && exit 1

DEVICE_COUNT=$(python3 -c "import torch_npu; print(torch_npu.npu.device_count())" 2>/dev/null || echo "8")
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
RESUME_STEPS=$((LATEST + 3))

cd "$VERL_DIR"
timeout 600 python3 -m verl.trainer.main_ppo \
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
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$RESUME_STEPS \
    trainer.default_local_dir=$CKPT_DIR &

TRAIN_PID=$!

echo "Waiting for step > $LATEST..."
for i in $(seq 1 60); do
    sleep 5
    NEW_STEP=$(find "$CKPT_SUBDIR" -maxdepth 1 -name "global_step_*" -type d 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
    if [ -n "$NEW_STEP" ] && [ "$NEW_STEP" -gt "$LATEST" ] 2>/dev/null; then
        echo "[OK] New checkpoint at step $NEW_STEP (was $LATEST)"
        kill $TRAIN_PID 2>/dev/null || true
        wait $TRAIN_PID 2>/dev/null || true
        echo "=== Checkpoint resume PASSED ==="
        exit 0
    fi
done

echo "[FAIL] No new checkpoint beyond step $LATEST within 5 minutes"
kill $TRAIN_PID 2>/dev/null || true
wait $TRAIN_PID 2>/dev/null || true
exit 1
