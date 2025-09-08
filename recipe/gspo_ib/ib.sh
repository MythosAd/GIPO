#!/bin/bash
#SBATCH --account=chenjun3
#SBATCH --partition=a100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
--mem=128G   # <--- 这是关键！请求 128GB 内存
# --- 准备工作 ---

# --- 环境配置 ---
# 切换到提交作业的目录，即 my_verl_project/
cd $SLURM_SUBMIT_DIR

# --- 使用 Singularity 执行主程序 ---
echo "Starting Singularity container and running the Python script..."

echo "Loading Singularity module..."
module purge
module load singularity/3.10  # 请确认版本号

# --- 执行完整的、单一的命令 ---
singularity exec \
    --nv \
    --bind ./verl:/app/verl \
    --bind ./recipe:/app/recipe \
    --bind ./model:/model \
    --bind ./ray_tmp:/tmp \
    --bind /proc:/proc \
    --env HF_HOME=/model/.cache \
    --env WANDB_MODE=offline \
    --env WANDB_DIR=/model \
    --env WANDB_API_KEY=5332e67da7daf0a41f52e3a3ed1377882ed276a2 \
    --env PYTHONPATH=/app \
    --env VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    --env RAY_memory_monitor_refresh_ms=0 \
    --env HYDRA_FULL_ERROR=1 \
    --env RAY_gcs_rpc_server_reconnect_timeout_s=0 \
    ./verl_image.sif \
    bash -c " \
        set -ex; \
        cd /app;  # 明确切换到 /app 目录
        pwd;      # 确认当前目录
        ls -la verl/trainer/config/ppo_trainer.yaml;  # 验证文件存在
        ulimit -n 51200; \
        ulimit -u 65535; \
        which python; \
        python -c 'import ray; print(f\"Ray version: {ray.__version__}\")'; \
        \
        echo '--- Starting main Python script ---'; \
        python -u -m  recipe.gspo_ib.main_ib \
            hydra.run.dir=/tmp/hydra_outputs \
            algorithm.adv_estimator=lonspo \
            actor_rollout_ref.actor.policy_loss.loss_mode='gspo' \
            actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean" \
            actor_rollout_ref.actor.clip_ratio_low=0.0003 \
            actor_rollout_ref.actor.clip_ratio_high=0.0004 \
            data.shuffle=True \
            data.prompt_key=prompt \
            data.truncation='error' \
            data.filter_overlong_prompts=true \
            data.train_batch_size=512 \
            data.max_prompt_length=2048 \
            data.max_response_length=8192 \
            actor_rollout_ref.rollout.n=16 \
            actor_rollout_ref.model.path=/model/Qwen2.5-0.5B \
            data.train_files=/model/data/gsm8k/train.parquet \
            data.val_files=/model/data/gsm8k/test.parquet \
            trainer.logger="['console','wandb']" \
            trainer.project_name='RL-GSPO' \
            trainer.experiment_name='gspo-lonspo-epslow-0.0003-epshigh-0.0004-Qwen2.5-0.5B-RL' \
            trainer.n_gpus_per_node=2 \
            trainer.nnodes=1 \
            trainer.val_before_train=False \
            trainer.test_freq=1 \
            trainer.save_freq=-1 \
            trainer.total_epochs=10 \
            trainer.total_training_steps=400 \
            trainer.default_local_dir=/app/verl/checkpoints \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
            actor_rollout_ref.actor.optim.weight_decay=0.1 \
            actor_rollout_ref.actor.ppo_mini_batch_size=128 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
            actor_rollout_ref.actor.grad_clip=1.0 \
            actor_rollout_ref.actor.entropy_coeff=0 \
            algorithm.use_kl_in_reward=False \
            algorithm.kl_ctrl.kl_coef=0.0 \
            actor_rollout_ref.actor.use_kl_loss=False \
            actor_rollout_ref.actor.kl_loss_coef=0.0 \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.use_dynamic_bsz=True \
            actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
            actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
            actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20480 \
            actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
            actor_rollout_ref.actor.fsdp_config.param_offload=True \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            actor_rollout_ref.actor.entropy_checkpointing=True \
            actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
            actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.mode=sync \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.enable_chunked_prefill=True \
            actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
            actor_rollout_ref.rollout.temperature=1.0 \
            actor_rollout_ref.rollout.top_p=1.0 \
            actor_rollout_ref.rollout.top_k=-1 \
            actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
            actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
            actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
            actor_rollout_ref.rollout.val_kwargs.do_sample=true \
            actor_rollout_ref.rollout.val_kwargs.n=1 \
            actor_rollout_ref.model.enable_activation_offload=True \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
            actor_rollout_ref.rollout.enforce_eager=True \
            actor_rollout_ref.rollout.free_cache_engine=True \
            reward_model.reward_manager=dapo \
            +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
            +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
            +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
            +reward_model.reward_kwargs.max_resp_len=8192 \
            custom_reward_function.path="recipe/gspo_ib/ib_reward_function.py" \
            custom_reward_function.name="compute_ib_composite_reward" \
            custom_reward_function.eta=1.0 \
            custom_reward_function.gamma=1.0 \
    "  > my_job_output.out 2>&1
	

# 检查退出码
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Job finished successfully."
else
    echo "Job failed with exit code $EXIT_CODE."
fi
