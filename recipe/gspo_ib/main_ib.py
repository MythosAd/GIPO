# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hydra
import ray

from .gspo_ib_ray_trainer import RayGSPOIBTrainer
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.trainer.ppo.reward import load_reward_manager


@hydra.main(config_path="config", config_name="gspo_ib_trainer", version_base=None)
def main(config):
    run_gspo_ib(config)


def run_gspo_ib(config, compute_score=None):
    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task.remote(config, compute_score))


@ray.remote(num_cpus=1)
def main_task(config, compute_score=None):
    from pprint import pprint
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf

    from verl.utils.fs import copy_local_path_from_hdfs

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(local_path)

    if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }

    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id


    
    # Load the reward manager for training and validation.
    reward_fn = load_reward_manager(
        config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
    )
    val_reward_fn = load_reward_manager(
        config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
    )
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayGSPOIBTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
