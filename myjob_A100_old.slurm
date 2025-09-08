#!/bin/bash
#SBATCH --account=chenjun3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
head_node_ip=$(hostname --ip-address)
# --- 1. 准备和打印作业信息 ---
echo "--- Preparing Job Environment ---"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo $pwd
cd $SLURM_SUBMIT_DIR || { echo "Failed to change directory to $SLURM_SUBMIT_DIR"; exit 1; }
mkdir -p slurm_logs

# --- 2. 加载主机模块 ---
echo "--- Loading host modules ---"
module purge
module load singularity/3.10
module load nvidia/cuda/12.2 # 确保这个和容器内的驱动兼容

# --- 3. 准备临时目录 ---
mkdir -p "ray_tmp"


# --- 4. 执行 Singularity 容器 ---
echo "--- Starting Singularity Container ---"
#-----------------------------------------------------------------#
singularity exec \
    ./verl_image.sif \
    ray stop


singularity exec \
    --nv \
    --bind ./verl:/app \
    --bind ./model:/model \
    --bind ./ray_tmp:/tmp \
    ./verl_image.sif \
    ray start --head --node-ip-address=$(hostname --ip-address) --port=6379 \
    --num-cpus 8 --num-gpus 1  --temp-dir=/tmp > ray_start.log 2>&1 &


### 移除软连接
mv /home/yangchengcao/.cache /home/yangchengcao/.cache_bak
### 显示这个目录下的文件 及大小
ls -la

# wandb sync
wandb sync wandb/offline-run-20250716_100000-example



singularity exec \
    --nv \
    --bind ./verl:/app \
    --bind ./model:/model \
    --bind ./ray_tmp:/tmp \
    --bind /home/yangchengcao/project/conda_cache:/home/yangchengcao/.cache \
    --env HF_HOME=/model/.cache \
    --env PYTHONPATH=/app \
    --env VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    --env RAY_memory_monitor_refresh_ms=0 \
    --env RAY_LOGGING_LEVEL=DEBUG \
    --env HYDRA_FULL_ERROR=1 \
    --env RAY_gcs_rpc_server_reconnect_timeout_s=60 \
    ./verl_image.sif \
    bash -c " \
        set -e; \
        echo '--- 开始执行路径诊断脚本 ---'; \
        python -u /app/verify_worker_paths.py \
    " 2>&1 | tee my_job_output.out

conda create -n verl python=3.10 -y
conda activate verl
tmux attach -t my_debug_session
du -sh /home/yangchengcao/.cache
tmux new -s my_debug_session
tmux attach -t my_debug_session
srun --account=chenjun3 --partition=a100x4 --nodes=1 --gres=gpu:1 --cpus-per-task=8 --mem=128G --pty /bin/bash
tmux new -s my_debug_session
tmux attach -t my_debug_session
srun --jobid=11842481 --pty nvidia-smi
srun --jobid=11865634 --pty nvidia-smi
module load singularity/3.10
module load nvidia/cuda/12.2
    --bind /home/yangchengcao/project:/home/yangchengcao/project \
    --bind /home/yangchengcao/.cache:/home/yangchengcao/.cache \
    --bind /home/yangchengcao/project/conda_cache:/home/yangchengcao/project/conda_cache \
singularity exec \
    --nv \
    --bind ./verl:/app \
    --bind ./model:/model \
    --bind ./ray_tmp:/tmp \
    --env HF_HOME=/model/.cache \
    --env WANDB_MODE=offline \
    --env WANDB_DIR=/model \
    --env WANDB_API_KEY=5332e67da7daf0a41f52e3a3ed1377882ed276a2 \
    --env PYTHONPATH=/app \
    --env VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    --env RAY_gcs_rpc_server_reconnect_timeout_s=60 \
    ./verl_image.sif \
    bash -c " \
        set -ex; \
        ulimit -n 51200; \
        ulimit -u 65535; \
        which python; \
        python -c 'import ray; print(f\"Ray version: {ray.__version__}\")'; \
        python -u -m recipe.gspo_ib.main_ib \
            data.shuffle=True \
            algorithm.adv_estimator=grpo \
            data.train_files=/model/data/gsm8k/train.parquet \
            data.val_files=/model/data/gsm8k/test.parquet \
            data.train_batch_size=1 \
            data.val_batch_size=1 \
            data.shuffle=True \
            data.prompt_key=prompt \
            data.truncation='error' \
            data.max_prompt_length=6144 \
            data.max_response_length=2048 \
            actor_rollout_ref.actor.policy_loss.loss_mode='gspo' \
            ctor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean" \
            actor_rollout_ref.actor.clip_ratio_low=0.0003 \
            actor_rollout_ref.actor.clip_ratio_high=0.0004 \
            actor_rollout_ref.model.path=/model/Qwen2.5-0.5B \
            actor_rollout_ref.actor.use_dynamic_bsz=True \
            actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
            actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
            actor_rollout_ref.actor.optim.weight_decay=0.1 \
            actor_rollout_ref.actor.ppo_mini_batch_size=16 \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.actor.use_kl_loss=False \
            actor_rollout_ref.actor.kl_loss_coef=0.0 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
            actor_rollout_ref.actor.grad_clip=1.0 \
            actor_rollout_ref.actor.entropy_coeff=0 \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=True \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.max_num_batched_tokens=14240 \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
            actor_rollout_ref.rollout.enforce_eager=False \
            actor_rollout_ref.rollout.free_cache_engine=True \
            actor_rollout_ref.rollout.enable_chunked_prefill=True \
            actor_rollout_ref.rollout.n=1 \
            actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
            actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
            actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
            actor_rollout_ref.rollout.val_kwargs.do_sample=true \
            actor_rollout_ref.rollout.val_kwargs.n=1 \
            actor_rollout_ref.rollout.temperature=1.0 \
            actor_rollout_ref.model.enable_activation_offload=True \
            actor_rollout_ref.rollout.temperature=1.0 \
            algorithm.use_kl_in_reward=False \
            algorithm.kl_ctrl.kl_coef=0.0 \
            trainer.critic_warmup=0 \
            trainer.logger=\"['wandb']\" \
            trainer.project_name='RL-GSPO-IB' \
            trainer.experiment_name='debug-gspo-lonspo-Qwen2.5-0.5B-RL' \
            trainer.n_gpus_per_node=2 \
            trainer.nnodes=1 \
            trainer.val_before_train=False \
            trainer.save_freq=1000 \
            trainer.total_epochs=10 \
            reward_model.reward_manager=gspoib \
            custom_reward_function.path=recipe/gspo_ib/reward_function.py \
            custom_reward_function.name=compute_score \
    "  2>&1 | tee my_job_output.out
	
# --- 5. 检查退出码并清理 ---
EXIT_CODE=$?
echo "--- Cleaning up temporary directory: ${MY_RAY_TMP_DIR} ---"
rm -rf ${MY_RAY_TMP_DIR}

if [ $EXIT_CODE -eq 0 ]; then
    echo "--- Job finished successfully. ---"
else
    echo "--- Job failed with exit code $EXIT_CODE. ---"
fi

exit $EXIT_CODE