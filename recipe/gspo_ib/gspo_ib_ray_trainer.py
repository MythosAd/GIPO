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

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
   reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.metric import reduce_metrics

class RayGSPOIBTrainer(RayPPOTrainer):
    """
    """
    def _create_rollout_gt_batch(self, batch: DataProto) -> DataProto:  
        # 从batch中提取rollout生成的响应和ground truth  
        responses = batch.batch["responses"]  # rollout生成的响应  
        ground_truths = batch.non_tensor_batch.get("ground_truth", [])  # 正确答案  

        # 拼接响应和正确答案，创建新的输入序列  
        # 具体实现取决于你的拼接策略  
        enhanced_sequences = self._concatenate_response_and_gt(responses, ground_truths)  

        # 创建新的DataProto用于重新推理  
        enhanced_batch = DataProto.from_dict({  
            "input_ids": enhanced_sequences,  
            "attention_mask": self._create_attention_mask(enhanced_sequences)  
        })

        return enhanced_batch


    def fit(self):
        """
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0
        
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
     
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        # ### 变量名优化 ###: `batch` -> `training_batch`，明确其作为最终训练批次的角色。
        training_batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                
                rollout_data: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                
                if "multi_modal_data" in rollout_data.non_tensor_batch.keys():
                    gen_batch = rollout_data.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = rollout_data.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):

                    # === 1. 生成 (Rollout) 阶段 ===
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    rollout_data.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(rollout_data.batch))], dtype=object
                    )
                    rollout_data = rollout_data.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    rollout_data = rollout_data.union(gen_batch_output)

                    # === 2. 计算 Log Probs ===
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(rollout_data)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = compute_response_mask(rollout_data)
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        metrics.update({"actor/entropy": entropy_agg.detach().item()})
                        old_log_prob.batch.pop("entropys")
                        rollout_data = rollout_data.union(old_log_prob)



                    with marked_timer("composite_reward", timing_raw, "yellow"):
                        # 1. external reward
                        try:
                            reward_result = self.reward_fn(rollout_data, return_dict=True)
                          
                            external_rewards = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            external_rewards = self.reward_fn(rollout_data)
                            reward_extra_infos_dict = {}

                        reasoning_token_lens = reward_extra_infos_dict.get("reasoning_token_len")
                        answer_token_lens = reward_extra_infos_dict.get("answer_token_len")
                        
                        composite_rewards = external_rewards


                        # 2. reasoning_entropy reward
                        batch_size = entropys.shape[0]
                        seq_len = entropys.shape[1]

                        start_indices = torch.argmax(response_masks.long(), dim=-1)

                        # 3. 计算推理和答案部分的结束索引
                        reasoning_end_indices = start_indices + torch.tensor(reasoning_token_lens, device=start_indices.device, dtype=start_indices.dtype)

                        answer_end_indices = reasoning_end_indices + torch.tensor(answer_token_lens, device=reasoning_end_indices.device, dtype=reasoning_end_indices.dtype)
                        seq_indices = torch.arange(seq_len, device=entropys.device, dtype=torch.int64).unsqueeze(0)
                        reasoning_mask = (seq_indices >= start_indices.unsqueeze(1)) & (seq_indices < reasoning_end_indices.unsqueeze(1))
                        answer_mask = (seq_indices >= reasoning_end_indices.unsqueeze(1)) & (seq_indices < answer_end_indices.unsqueeze(1))
                        
                        decay_rate = self.config.algorithm.get("entropy_decay_rate",1.0)
                        seq_len = entropys.shape[1]
                        
                        reasoning_start_indices = torch.argmax(reasoning_mask.long(), dim=-1, keepdim=True)
                        relative_indices = torch.clamp(seq_indices - reasoning_start_indices, min=0)
                        position_weights = torch.pow(decay_rate, relative_indices)
                        
                        reasoning_reward =   (entropys * position_weights) * reasoning_mask

                        eta = self.config.algorithm.get("eta", 0.1)
 
                        # 在现有的奖励计算之后添加  
                        # 3. answer_log_prob reward
                        with marked_timer("rollout_with_gt_inference", timing_raw, color="purple"):  
                            # 拼接rollout结果和正确答案  
                            # enhanced_batch = self._create_rollout_gt_batch(batch)  
                            
                            responses = rollout_data.batch["responses"]  # rollout生成的响应   8 x 512   151643 padding
                            
                            # 收集 ground-truth 文本（最小化临时对象）
                            ground_truth_ids_list_of_tensors = [torch.tensor(rollout_data[i].non_tensor_batch["reward_model"]["ground_truth_ids"], dtype=torch.int64) for i in range(len(responses))]
                            
                            
                            # 2. 使用 pad_sequence 进行填充和堆叠
                            # batch_first=True 使输出的维度为 (batch_size, max_length)
                            # padding_value=0 指定用 0 进行填充,其实是有tokenizer的。 init有。
                            # ground_truth_tensor = torch.nn.utils.rnn.pad_sequence(ground_truth_ids_list_of_tensors, batch_first=True, padding_value=0)
                            # 拼接响应和正确答案，创建新的输入序列
                            # 具体实现取决于你的拼接策略
                            
                            reasoning_len = reasoning_mask.sum(-1)
                            new_sequences_list=[]
                            
                            for i in range(responses.shape[0]):
                                response_i = responses[i]
                                
                                ground_truth_i = ground_truth_ids_list_of_tensors[i]
                               
                                # 2. 获取截断后的 response token
                                truncated_response = response_i[:reasoning_len[i]]
                                
                                # 3. 拼接成最终想要的序列
                                new_sequence = torch.cat([truncated_response, ground_truth_i])
                                new_sequences_list.append(new_sequence)

                            # 7. 对新的、长度不一的序列列表进行填充
                            pad_token_id = self.tokenizer.pad_token_id
                            enhanced_sequences = torch.nn.utils.rnn.pad_sequence(
                                new_sequences_list, 
                                batch_first=True, 
                                padding_value=pad_token_id
                            )

                            # 8. 基于最终的序列，创建非常简单的 attention mask
                            reasoning_answer_attention_mask = (enhanced_sequences != pad_token_id).long()
                            position_ids = torch.arange(enhanced_sequences.size(1)).unsqueeze(0).expand(enhanced_sequences.size(0), -1)

                            # 创建新的DataProto用于重新推理
                            enhanced_batch = DataProto.from_dict({  
                                "input_ids": enhanced_sequences,   
                                "attention_mask": reasoning_answer_attention_mask,  
                                "responses": enhanced_sequences, 
                                "position_ids": position_ids
                            })
                            # 重新推理计算log概率作为内部奖励  
                            reasoning_answer_log_probs = self.actor_rollout_wg.compute_log_prob(enhanced_batch) 

                            #  建立 reasoning_attention_mask
                            ground_answer_mask = torch.zeros(
                                enhanced_sequences.shape[0], 
                                enhanced_sequences.shape[1] - 1, 
                                device=enhanced_sequences.device
                            )

                            for i in range(ground_answer_mask.shape[0]):
                                ground_truth_i =ground_truth_ids_list_of_tensors[i]
                                reasoning_len
                                ground_truth_i_len =ground_truth_i.shape[0]
                                ground_answer_mask[i,  reasoning_len[i]-1 : reasoning_len[i]-1 + ground_truth_i_len] = 1
                            
                            answer_reward = eta * reasoning_answer_log_probs.batch['old_log_probs'] * ground_answer_mask
                            
                            # 4. 计算复合奖励 仔细想想怎么计算合适  先使用标量实验.
                            reasoning_r= reasoning_reward.sum(-1)
                            answer_r =answer_reward.sum(-1)
                            id2reasoning_score = defaultdict(list)
                            id2answer_score = defaultdict(list)
                            id2reasoning_mean = {}
                            id2reasoning_std = {}
                            id2answer_mean = {}
                            id2answer_std = {}
                            normalized_reasoning_scores = []
                            normalized_answer_scores = []
                            index = rollout_data.non_tensor_batch["uid"]
                            with torch.no_grad():
                                bsz = reasoning_r.shape[0]
                                for i in range(bsz):
                                    id2reasoning_score[index[i]].append(reasoning_r[i])
                                    id2answer_score[index[i]].append(answer_r[i])
                                for idx in id2reasoning_score:
                                    if len(id2reasoning_score[idx]) == 1:
                                        id2reasoning_mean[idx] = torch.tensor(0.0)
                                        id2reasoning_std[idx] = torch.tensor(1.0)

                                        id2answer_mean[idx] = torch.tensor(0.0)
                                        id2answer_std[idx] = torch.tensor(1.0)
                                    elif len(id2reasoning_score[idx]) > 1:
                                        reasoningscores_tensor = torch.stack(id2reasoning_score[idx])

                                        id2reasoning_mean[idx] = torch.mean(reasoningscores_tensor)
                                        id2reasoning_std[idx] = torch.std(reasoningscores_tensor)

                                        answerscores_tensor = torch.stack(id2answer_score[idx])

                                        id2answer_mean[idx] = torch.mean(answerscores_tensor)
                                        id2answer_std[idx] = torch.std(answerscores_tensor)
                                    else:
                                        raise ValueError(f"no score in prompt index: {idx}")
                                for i in range(bsz):
                                    # if norm_adv_by_std_in_grpo:
                                        normalized_reasoning_score = (reasoning_r[i] - id2reasoning_mean[index[i]]) / (id2reasoning_std[index[i]] + 1e-6)
                                        normalized_answer_score = (answer_r[i] - id2answer_mean[index[i]]) / (id2answer_std[index[i]] + 1e-6)
                                        normalized_reasoning_scores.append(normalized_reasoning_score)
                                        normalized_answer_scores.append(normalized_answer_score)
                                    # else:
                                    #     scores[i] = scores[i] - id2mean[index[i]]

                            reasoning_r_regular = torch.stack(normalized_reasoning_scores)
                            answer_r_regular = torch.stack(normalized_answer_scores)



                            gamma = self.config.algorithm.get("gamma", 0.01)
                            eta = self.config.algorithm.get("eta", 0.01)
                            breakpoint()
                            intrinsic_reward = reasoning_r_regular*gamma + answer_r_regular*eta # batch_size x response_len
                            
                            # 创建与external_rewards相同形状的零张量  
                            
                            # 找到每个序列的最后一个有效token位置  
                            last_token_indices =  (responses != pad_token_id).long().sum(-1)-1 # batch_size  

                            intrinsic_reward_expanded = torch.zeros_like(external_rewards)  # batch_size x seq_len  
                            # 将intrinsic_reward分配到最后一个有效token位置  
                            batch_indices = torch.arange(intrinsic_reward.size(0))  
                            intrinsic_reward_expanded[batch_indices, last_token_indices] = intrinsic_reward.squeeze(-1)  
                            
                            composite_rewards = external_rewards + intrinsic_reward_expanded
                            
                            metrics.update({
                                "reward/reasoning_r": reasoning_r.mean().item(),
                                "reward/answer_r": answer_r.mean().item(),
                            })

                        rollout_data.batch["token_level_scores"] = composite_rewards
                        rollout_data.batch["token_level_rewards"] = composite_rewards

                    # === 3. 样本过滤与批次累积 ===
                    if not self.config.algorithm.filter_groups.enable:
                        training_batch = rollout_data
                    else:
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_reward":
                             rollout_data.non_tensor_batch["seq_reward"] = (
                                rollout_data.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            rollout_data.non_tensor_batch["uid"], rollout_data.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {uid: np.std(vals) for uid, vals in prompt_uid2metric_vals.items()}
                        kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0 or len(prompt_uid2metric_vals[uid]) == 1]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = [idx for idx, uid in enumerate(rollout_data.non_tensor_batch["uid"]) if uid in kept_prompt_uids]
                        
                        filtered_rollout_data = rollout_data[kept_traj_idxs]
                        training_batch = filtered_rollout_data if training_batch is None else DataProto.concat([training_batch, filtered_rollout_data])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                progress_bar.update(1)
                                self.gen_steps += 1
                                continue
                            else:
                                raise ValueError(f"Generated too many batches ({num_gen_batches}) without filling a training batch. Check data quality or increase max_num_gen_batches.")
                        else:
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            training_batch = training_batch[:traj_bsz]

                    # === 4. PPO 更新阶段 ===
                    training_batch.batch["response_mask"] = compute_response_mask(training_batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(training_batch, metrics=metrics)

                    training_batch.meta_info["global_token_num"] = torch.sum(training_batch.batch["attention_mask"], dim=-1).tolist()
                    
                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(training_batch)
                            training_batch = training_batch.union(ref_log_prob)
        
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(training_batch)
                            training_batch = training_batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        training_batch = compute_advantage(
                            training_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # === 5. 模型更新 ===
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(training_batch)
                        metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(training_batch)
                        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                    # === 6. 验证与保存 ===
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

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # === 7. 日志记录与清理 ===
                metrics.update(compute_data_metrics(batch=training_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=training_batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=training_batch, timing_raw=timing_raw, n_gpus=n_gpus))
                
                timing_raw = defaultdict(float)
                metrics["train/num_gen_batches"] = num_gen_batches
                training_batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
                
                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1