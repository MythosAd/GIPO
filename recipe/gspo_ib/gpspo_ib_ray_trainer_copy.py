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

import uuid  # 用于生成唯一的ID
from collections import defaultdict  # 用于创建默认值为特定类型的字典
from copy import deepcopy  # 用于创建对象的深拷贝
from pprint import pprint  # 用于美观地打印复杂的数据结构

import numpy as np  # 导入 NumPy 库
import torch  # 导入 PyTorch 库
from tqdm import tqdm  # 导入 tqdm 库，用于显示进度条

from verl import DataProto  # 导入自定义的数据协议类
from verl.trainer.ppo.core_algos import agg_loss  # 从核心算法模块导入损失聚合函数
from verl.trainer.ppo.metric_utils import ( # 从指标工具模块导入一系列计算函数
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (  # 从 Ray 训练器基类模块导入相关组件
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer  # 导入一个计时器上下文管理器
from verl.utils.model import compute_position_id_with_mask  # 导入计算 position_id 的工具函数


class RayGSPOIBTrainer(RayPPOTrainer):
    """
    GSPO-IB (Goal-driven Self-improving Policy Optimization with In-context Bootstrapping) 的 Ray 训练器。
    这个训练器实现了 GSPO-IB 算法，该算法旨在通过一种复合奖励机制来提升需要复杂推理的任务（如数学问题）的性能。
    """

    def fit(self):
        """
        GSPO-IB 算法的核心训练循环。
        这个循环负责协调数据生成、复合奖励计算、优势估计和模型更新等步骤。
        """
        from omegaconf import OmegaConf  # 导入 OmegaConf 库，用于处理复杂的配置

        from verl.utils.tracking import Tracking  # 导入追踪/日志工具

        # 初始化日志记录器
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # 初始化全局步数和生成步数
        self.global_steps = 0
        self.gen_steps = 0
        # 尝试从检查点加载模型，恢复训练
        breakpoint()  
        # self._load_checkpoint() 
        # 在训练前先做一次验证（如果配置允许）
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate() # 调用验证函数
            assert val_metrics, f"{val_metrics=}" # 确保验证有返回结果
            pprint(f"Initial validation metrics: {val_metrics}") # 打印初始验证指标
            logger.log(data=val_metrics, step=self.global_steps) # 记录初始验证指标
            # 如果配置为"只验证不训练"，则在此处直接返回
            if self.config.trainer.get("val_only", False):
                return

        # 创建训练进度条
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        
        # 我们从第 1 步开始训练
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None # 用于存储最后一次的验证指标

        # --- 性能分析 (Profiling) 相关的标志位 ---
        # 上一步是否进行了性能分析
        prev_step_profile = False
        # 当前步是否需要进行性能分析 (根据配置文件中的 `profile_steps` 列表)
        curr_step_profile = True
        # 下一步是否需要进行性能分析
        next_step_profile = False

        # 初始化一个字典来记录各个阶段的耗时
        timing_raw = defaultdict(float)
        # 初始化 `batch` 为 None，用于后续累积数据
        batch = None
        # 当前累积的 batch 中包含的 prompt 数量
        num_prompt_in_batch = 0
        # 当前为一个训练步已经生成了多少个批次的数据（在启用过滤时使用）
        num_gen_batches = 0

        # 外层循环：遍历所有 epoch
        for epoch in range(self.config.trainer.total_epochs):
            # 内层循环：遍历训练数据加载器中的每个批次
            for batch_dict in self.train_dataloader:
                metrics = {} # 初始化一个空字典，用于收集当前步骤的所有指标
                # 将从 dataloader 中获取的字典转换为自定义的 DataProto 对象
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # 弹出（pop）生成（generation）阶段需要的键
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    # 如果是多模态数据，需要额外 pop 出 'multi_modal_data'
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    # 对于纯文本数据，只 pop 出文本相关的键
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                
                # 为了增加探索，对每个 prompt 生成 n 个不同的响应（rollout）
                # 这里通过 `repeat` 方法将 gen_batch 复制 n 份
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                # 判断当前是否为整个训练过程的最后一步
                is_last_step = self.gen_steps >= self.total_training_steps

            

                # 使用计时器记录整个 PPO 单步（step）的耗时
                with marked_timer("step", timing_raw):
                    # === 1. 生成 (Rollout) 阶段 ===
                    # 使用计时器记录生成响应的耗时
                    with marked_timer("gen", timing_raw, "red"):
                        # 调用 actor worker group 的远程方法来生成序列
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                    # 为每条数据生成一个唯一ID，用于后续在乱序后也能正确匹配数据
                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # 将 prompt 数据也复制 n 份，与生成的响应对齐
                    batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # 将生成结果（如序列）合并到主批次中
                    batch = batch.union(gen_batch_output)
                    # 计算响应部分的 mask，用于后续计算中忽略 prompt 部分
                    if "response_mask" not in batch.batch.keys():# batch.batch["attention_mask"].shape
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    
                    # （可选）平衡不同 GPU 卡上的 token 数量，以优化数据并行（DP）的负载均衡
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # 计算全局有效 token 数量
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # 4. 【计算旧策略的 Log Probs】
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        # 计算生成响应时所用的策略（即更新前的 Actor）的对数概率
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        # 提取每个 token 的对数概率和熵   # shape: (batch_size, response_len)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                        
              
                    # === 2. GSPO-IB 复合奖励 (Composite Reward) 计算阶段 ===
                    with marked_timer("composite_reward", timing_raw, "yellow"):
                        try:
                            reward_result = self.reward_fn(batch, return_dict=True)
                            # scalar_rewards 是外部奖励，已经是序列级的
                            # 形状为 [B, L]，但在 response 最后一个 token 才有值
                            scalar_rewards = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            scalar_rewards = self.reward_fn(batch)
                            reward_extra_infos_dict = {}

                        reasoning_token_lens = reward_extra_infos_dict.get("reasoning_token_len")
                        answer_token_lens = reward_extra_infos_dict.get("answer_token_len")


                        # 初始化最终奖励，默认等于外部奖励
                        final_rewards = scalar_rewards

                         # === 3. 优势 (Advantage) & 价值 (Value) 计算 ===
                        norm_adv_by_std_in_grpo=True
                        scores = final_rewards.sum(dim=-1)
                        epsilon: float = 1e-6
                        id2score = defaultdict(list)
                        id2mean = {}
                        id2std = {}
                        index = batch.non_tensor_batch["uid"]
                        response_mask = batch.batch["response_mask"]
                        with torch.no_grad():
                            bsz = scores.shape[0]
                            for i in range(bsz):
                                id2score[index[i]].append(scores[i])
                            for idx in id2score:
                                if len(id2score[idx]) == 1:
                                    id2mean[idx] = torch.tensor(0.0)
                                    id2std[idx] = torch.tensor(1.0)
                                elif len(id2score[idx]) > 1:
                                    scores_tensor = torch.stack(id2score[idx])
                                    id2mean[idx] = torch.mean(scores_tensor)
                                    id2std[idx] = torch.std(scores_tensor)
                                else:
                                    raise ValueError(f"no score in prompt index: {idx}")
                            for i in range(bsz):
                                if norm_adv_by_std_in_grpo:
                                    scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
                                else:
                                    scores[i] = scores[i] - id2mean[index[i]]
                            scores = scores.unsqueeze(-1) * response_mask

    
                        
                        if reasoning_token_lens and answer_token_lens:
                            batch_size = entropys.shape[0]

                            reasoning_mask = torch.zeros_like(entropys, dtype=torch.bool)
                            answer_mask = torch.zeros_like(entropys, dtype=torch.bool)

                            for i in range(batch_size):
                                len_r = reasoning_token_lens[i]
                                len_a = answer_token_lens[i]

                                # 找出响应的起始位置 (第一个为 True 的地方)
                                response_start_indices = torch.where(response_masks[i])[0]
                                if len(response_start_indices) == 0:
                                    continue # 该序列没有响应
                                start_index = response_start_indices[0]

                                # 从响应的起始位置开始创建 mask
                                reasoning_mask[i, start_index : start_index + len_r] = True
                                answer_mask[i, start_index + len_r : start_index + len_r + len_a] = True

                            final_reasoning_mask = reasoning_mask & response_masks.bool()
                            final_answer_mask = answer_mask & response_masks.bool()

                            # ### 核心修正：计算并应用分布式的内在奖励 ###

                            # --- 1. 思维多样性奖励 (Token-level, 修正后) ---
                            # 目标: 鼓励模型在推理阶段探索更多样化的思考路径，并按 token 分布奖励。
                            # 方法: 将每个 reasoning token 的熵作为奖励，并施加一个从 reasoning 部分开始随位置递减的权重。
                            gamma = self.config.algorithm.get("gamma_entropy", 0.01)
                            decay_rate = self.config.algorithm.get("entropy_decay_rate", 0.999) # 1.0 表示不衰减
                            batch_size, seq_len = entropys.shape

                            # --- 创建相对于 reasoning 起始位置的衰减权重 ---
                            seq_indices = torch.arange(seq_len, device=entropys.device, dtype=torch.float32).unsqueeze(0)
                            reasoning_start_indices = torch.argmax(final_reasoning_mask.long(), dim=-1, keepdim=True)
                            relative_indices = seq_indices - reasoning_start_indices
                            relative_indices[relative_indices < 0] = 0
                            position_weights = torch.pow(decay_rate, relative_indices)

                            weighted_entropies = entropys * position_weights
                            reasoning_reward = gamma * weighted_entropies * final_reasoning_mask

                            # --- 2. 推理质量奖励 (Token-level) ---
                            # 目标: 鼓励模型在给出推理后，生成高质量、高置信度的答案。
                            # 方法: 将每个 answer token 的对数概率作为奖励。
                            eta = self.config.algorithm.get("eta", 0.1)
                            log_probs = batch.batch["old_log_probs"]
                            answer_reward = eta * log_probs * final_answer_mask

                            # --- 3. 合并与应用奖励 ---
                            token_level_intrinsic_reward = (reasoning_reward + answer_reward).detach()
                            final_rewards = scores + token_level_intrinsic_reward
                            
                            # --- 4. 指标记录 ---
                            reasoning_total_entropy = (entropys * final_reasoning_mask).sum(dim=-1)
                            answer_total_log_prob = (log_probs * final_answer_mask).sum(dim=-1) # 使用 log_probs
                            composite_ib_reward = token_level_intrinsic_reward.sum(dim=-1)

                            last_token_indices = response_masks.sum(dim=-1).long() - 1
                            metrics.update({"reward/scalar_rewards": scalar_rewards[torch.arange(batch_size), last_token_indices].mean().item()})
                            metrics.update({
                                "reward/reasoning_total_entropy": reasoning_total_entropy.mean().item(),
                                "reward/answer_total_log_prob": answer_total_log_prob.mean().item(),
                                "reward/composite_ib_reward": composite_ib_reward.mean().item(),
                                "reward/token_level_intrinsic_reward": token_level_intrinsic_reward.mean().item(),
                            })

                        # 将最终的、正确的、序列级的复合奖励赋值给 `batch`
                        batch.batch["token_level_rewards"] = final_rewards # t
                        batch.batch['token_level_scores'] = final_rewards

                    # === 3. 优势 (Advantage) & 价值 (Value) 计算 ===
                    # with marked_timer("adv", timing_raw, "brown"):
                    #     # 现在，`compute_grpo_outcome_advantage` 会收到一个正确的、序列级的奖励张量
                    #     # `scalar_rewards` 仍然是稀疏的，但其在最后一个 token 上的值现在包含了我们所有的奖励信号
                    #     norm_adv_by_std_in_grpo=True
                    #     scores = batch.batch["token_level_rewards"].sum(dim=-1)
                    #     epsilon: float = 1e-6
                    #     id2score = defaultdict(list)
                    #     id2mean = {}
                    #     id2std = {}
                    #     index = batch.non_tensor_batch["uid"]
                    #     response_mask = batch.batch["response_mask"]
                    #     with torch.no_grad():
                    #         bsz = scores.shape[0]
                    #         for i in range(bsz):
                    #             id2score[index[i]].append(scores[i])
                    #         for idx in id2score:
                    #             if len(id2score[idx]) == 1:
                    #                 id2mean[idx] = torch.tensor(0.0)
                    #                 id2std[idx] = torch.tensor(1.0)
                    #             elif len(id2score[idx]) > 1:
                    #                 scores_tensor = torch.stack(id2score[idx])
                    #                 id2mean[idx] = torch.mean(scores_tensor)
                    #                 id2std[idx] = torch.std(scores_tensor)
                    #             else:
                    #                 raise ValueError(f"no score in prompt index: {idx}")
                    #         for i in range(bsz):
                    #             if norm_adv_by_std_in_grpo:
                    #                 scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
                    #             else:
                    #                 scores[i] = scores[i] - id2mean[index[i]]
                    #         scores = scores.unsqueeze(-1) * response_mask
                        batch.batch["advantages"] = final_rewards
                        batch.batch["returns"] = final_rewards
    
                    # === 4. 模型更新 ===
                    # 更新 Critic 模型
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                            metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

                    # 在 Critic 预热结束后，开始更新 Actor 模型
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                            metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)



                # === 5. 日志记录与检查点 ===
                # 收集与数据、时间、吞吐量相关的指标
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # 记录所有指标
                logger.log(data=metrics, step=self.global_steps)
                if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                    ):

                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()
                # 判断是否已完成所有训练步骤
                if self.global_steps >= self.total_training_steps:
                    progress_bar.close() # 关闭进度条
                    return
                
                # 更新进度条和步数
                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1