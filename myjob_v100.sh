#!/bin/bash
# --- 准备工作 ---

# --- 环境配置 ---

# --- 使用 Singularity 执行主程序 ---
echo "Starting Singularity container and running the Python script..."

echo "Loading Singularity module..."
module purge
module load singularity/3.10  # 请确认版本号

# 设置最大重试次数  
MAX_RETRIES=20
RETRY_COUNT=0  
  
# 设置最大重试次数  
MAX_RETRIES=20
RETRY_COUNT=0  
ulimit -n 51200; \
ulimit -u 65535; \
# 从之前的日志中提取wandb run ID（如果存在）  
WANDB_RUN_ID=""  
JOB_WANDB_RUN_ID="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
echo "This job will use a fixed wandb run ID: ${JOB_WANDB_RUN_ID}"
  
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do  
    echo "--- Attempt $((RETRY_COUNT + 1)) of $MAX_RETRIES ---"
    # 执行训练命令  
    # --- 执行完整的、单一的命令 ---
    singularity exec \
        --nv \
        --bind ./verl:/app/verl \
        --bind ./model:/model \
        --bind ./ray_tmp:/tmp \
        --env HF_HOME=/model/.cache \
        --env WANDB_MODE=offline \
        --env WANDB_DIR=/model \
        --env WANDB_API_KEY=5332e67da7daf0a41f52e3a3ed1377882ed276a2 \
        --env WANDB_RUN_ID="${JOB_WANDB_RUN_ID}" \
        --env WANDB_NAME="${JOB_WANDB_RUN_ID}" \
        --env PYTHONPATH=/app \
        --bind ./recipe:/app/recipe \
        --env VLLM_ATTENTION_BACKEND=FLASH_ATTN \
        ./verl_image.sif \
        bash -c " \
            set -ex; \
            ulimit -n 51200; \
            ulimit -u 65535; \
            which python; \
            python -c 'import ray; print(f\"Ray version: {ray.__version__}\")'; \
            python -u -m recipe.gspo_ib.main_ib \
                algorithm.adv_estimator=grpo \
                actor_rollout_ref.actor.policy_loss.loss_mode='gspo' \
                actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean" \
                actor_rollout_ref.actor.clip_ratio_low=0.0003 \
                actor_rollout_ref.actor.clip_ratio_high=0.0004 \
                actor_rollout_ref.actor.use_dynamic_bsz=True \
                actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.actor.use_kl_loss=False \
                actor_rollout_ref.actor.kl_loss_coef=0.0 \
                actor_rollout_ref.actor.kl_loss_type=low_var_kl \
                actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
                actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
                actor_rollout_ref.actor.grad_clip=1.0 \
                actor_rollout_ref.actor.entropy_coeff=0 \
                actor_rollout_ref.actor.optim.lr=1e-6 \
                actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
                actor_rollout_ref.actor.optim.weight_decay=0.1 \
                actor_rollout_ref.actor.ppo_mini_batch_size=16 \
                actor_rollout_ref.actor.fsdp_config.param_offload=false \
                actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
                actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
                actor_rollout_ref.actor.strategy="fsdp2" \
                actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
                actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
                actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
                actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
                actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
                actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
                actor_rollout_ref.model.path=/model/Qwen2.5-0.5B \
                actor_rollout_ref.model.use_remove_padding=True \
                actor_rollout_ref.model.enable_gradient_checkpointing=True \
                actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
                actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
                actor_rollout_ref.rollout.name=vllm \
                actor_rollout_ref.rollout.max_num_batched_tokens=14240 \
                actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
                actor_rollout_ref.rollout.enforce_eager=False \
                actor_rollout_ref.rollout.free_cache_engine=True \
                actor_rollout_ref.rollout.enable_chunked_prefill=True \
                actor_rollout_ref.rollout.mode=sync \
                actor_rollout_ref.rollout.n=8 \
                actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
                actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
                actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
                actor_rollout_ref.rollout.multi_stage_wake_up=true \
                actor_rollout_ref.rollout.val_kwargs.do_sample=true \
                actor_rollout_ref.rollout.val_kwargs.n=1 \
                actor_rollout_ref.rollout.temperature=1.0 \
                actor_rollout_ref.model.enable_activation_offload=True \
                actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
                actor_rollout_ref.rollout.temperature=1.0 \
                algorithm.use_kl_in_reward=False \
                algorithm.kl_ctrl.kl_coef=0.0 \
                data.shuffle=True \
                data.train_files=/model/data/gsm8k/train.parquet \
                data.val_files=/model/data/gsm8k/test.parquet \
                data.train_batch_size=512 \
                data.shuffle=True \
                trainer.test_freq=20 \
                data.prompt_key=prompt \
                data.truncation='error' \
                data.max_prompt_length=2048 \
                data.max_response_length=8192 \
                trainer.resume_mode=auto \
                trainer.critic_warmup=0 \
                trainer.logger=\"['wandb']\" \
                trainer.project_name='RL-GSPO-IB' \
                trainer.experiment_name='debug-gspo-lonspo-Qwen2.5-0.5B-RL' \
                trainer.n_gpus_per_node=2 \
                trainer.default_local_dir=/app/verl/checkpoints \
                trainer.nnodes=1 \
                trainer.val_before_train=True \
                trainer.save_freq=1 \
                trainer.total_epochs=10 \
                reward_model.reward_manager=gspoib \
                custom_reward_function.path=/app/recipe/gspo_ib/reward_function.py \
                custom_reward_function.name=compute_score \
        "  2>&1 | tee -a my_job_output.out

      
    if grep -q -E '\b([0-9]+)/\1\b' my_job_output.out; then
        echo "--- Training successfully completed (found completion pattern in log). ---"
        echo "--- Job finished after $((RETRY_COUNT + 1)) attempts. ---"
        exit 0
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))  
      
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then  
        echo "--- Job failed with exit code $EXIT_CODE. Retrying in 30 seconds... ---"  
        sleep 30  
          
        # 清理Ray进程  
        ray stop --force || true  
          
        # 清理可能的僵尸进程  
        pkill -f "python.*recipe.gspo_ib.main_ib" || true  
        sleep 10  
    else  
        echo "--- Job failed after $MAX_RETRIES attempts with exit code $EXIT_CODE. ---"  
        exit 1 # 使用非零退出码表示失败 
    fi  
done