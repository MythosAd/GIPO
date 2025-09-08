#!/bin/bash

#SBATCH --account=chenjun3
#SBATCH --partition=a100x4

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.error
#SBATCH --output=SBATCH.out


cd $SLURM_SUBMIT_DIR
module load nvidia/cuda/12.2

export N_GPUS=4
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo
data.train_files=/project/yangchengcao/conda_cache/huggingface/hub/datasets--openai--gsm8k/snapshots/e53f048856ff4f594e959d75785d2c2d37b678ee/main/train-00000-of-00001.parquet
data.val_files=/project/yangchengcao/conda_cache/huggingface/hub/datasets--openai--gsm8k/snapshots/e53f048856ff4f594e959d75785d2c2d37b678ee/main/test-00000-of-00001.parquet
data.train_batch_size=256
data.val_batch_size=1312
data.max_prompt_length=256
data.max_response_length=1024
actor_rollout_ref.model.path=/project/yangchengcao/conda_cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.ppo_mini_batch_size=128
actor_rollout_ref.actor.ppo_micro_batch_size=8
actor_rollout_ref.rollout.log_prob_micro_batch_size=8
actor_rollout_ref.rollout.tensor_model_parallel_size=2
actor_rollout_ref.rollout.gpu_memory_utilization=0.4
actor_rollout_ref.ref.log_prob_micro_batch_size=4
critic.optim.lr=1e-5
critic.model.path=/project/yangchengcao/conda_cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
critic.ppo_micro_batch_size=8
algorithm.kl_ctrl.kl_coef=0.001
trainer.logger=['wandb']
+trainer.val_before_train=False
trainer.default_hdfs_dir=null
trainer.n_gpus_per_node=1
trainer.nnodes=1
trainer.save_freq=100
trainer.test_freq=100
trainer.project_name=TinyZero
trainer.experiment_name=countdown-qwen2.5-1b
trainer.total_epochs=15

