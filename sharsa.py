import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoModel, AutoTokenizer, logging as hf_logging
import numpy as np
import copy
from typing import List, Dict

# 忽略transformers加载模型时的部分警告信息
hf_logging.set_verbosity_error()

# --- 0. 全局配置 ---
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LLM_MODEL_NAME = "distilbert-base-uncased"
    LR_ACTOR_CRITIC = 1e-4
    GAMMA = 0.99
    K_EPOCHS = 5
    EPS_CLIP = 0.2
    BATCH_SIZE = 2
    MAX_EPISODES = 100
    UPDATE_AFTER_EPISODES = 2 # 每生成2个episode后进行一次更新

# --- 1. 模拟环境和奖励模型 ---
class MockEnv:
    """
    一个模拟的文本生成环境。
    """
    def __init__(self):
        # 我们的目标是让LLM学会分两步生成这个目标句子
        self.target_text = "The quick brown fox jumps over the lazy dog."
        self.target_part1 = "The quick brown fox"
        self.target_part2 = " jumps over the lazy dog."
        self.plan = ["生成第一部分", "生成第二部分"]
        self.prompt = "任务：请生成一个描述动物的完整句子。"
        
    def get_plan(self):
        return self.plan

    def get_prompt(self):
        return self.prompt

class RewardModel:
    """
    一个模拟的奖励模型。
    """
    def __init__(self, target_text):
        self.target_text = target_text
        self.tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL_NAME)

    def evaluate(self, generated_text: str) -> float:
        """
        奖励基于生成文本和目标文本的相似度（这里用token重叠度简化）。
        """
        target_tokens = set(self.tokenizer.tokenize(self.target_text))
        generated_tokens = set(self.tokenizer.tokenize(generated_text))
        
        if not generated_tokens: return 0.0
        
        intersection = target_tokens.intersection(generated_tokens)
        union = target_tokens.union(generated_tokens)
        
        # Jaccard Similarity as reward
        reward = len(intersection) / len(union)
        return reward

# --- 2. 模型定义 ---
class ActorCriticLLM(nn.Module):
    """
    共享一个LLM backbone的Actor-Critic模型
    """
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        
        # 加载共享的LLM backbone
        self.backbone = AutoModel.from_pretrained(Config.LLM_MODEL_NAME)
        self.hidden_dim = self.backbone.config.hidden_size
        
        # Actor Head (策略头)：用于生成下一个token的概率分布
        self.actor_head = nn.Linear(self.hidden_dim, self.tokenizer.vocab_size)
        
        # Critic Head (价值头)：用于评估当前状态的价值
        self.critic_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        前向传播，返回token概率和状态价值
        """
        # 通过backbone获取最后一个token的hidden state
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state[:, -1, :] # (batch_size, hidden_dim)
        
        # 计算Actor输出 (logits for next token)
        action_logits = self.actor_head(last_hidden_state)
        
        # 计算Critic输出 (state value)
        state_value = self.critic_head(last_hidden_state)
        
        return action_logits, state_value

# --- 3. PPO Agent ---
class LLM_SHARSA_Agent:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL_NAME)
        
        # 实例化新旧两个策略网络
        self.policy = ActorCriticLLM(self.tokenizer).to(Config.DEVICE)
        self.policy_old = ActorCriticLLM(self.tokenizer).to(Config.DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=Config.LR_ACTOR_CRITIC)
        self.MseLoss = nn.MSELoss()
        
        # 存储rollout数据的缓冲区
        self.rollouts: List[List[Dict]] = []

    def generate_segment(self, current_text: str, subgoal: str, add_exploration=True, max_len=10) -> (str, float):
        """
        低层执行：根据当前文本和子目标，生成一小段文本。
        返回 (生成的片段, 这段生成的log_prob总和)
        """
        # 将子目标融入prompt，引导生成
        prompt_with_subgoal = f"{current_text} [SUBGOAL: {subgoal}]"
        
        input_ids = self.tokenizer.encode(prompt_with_subgoal, return_tensors='pt').to(Config.DEVICE)
        
        generated_ids = []
        total_log_prob = 0
        
        for _ in range(max_len):
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                logits, _ = self.policy_old(input_ids, attention_mask)
            
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            
            action_token_id = dist.sample() if add_exploration else torch.argmax(probs, dim=-1)
            
            # 检查是否生成结束符
            if action_token_id.item() == self.tokenizer.sep_token_id:
                break
                
            generated_ids.append(action_token_id.item())
            total_log_prob += dist.log_prob(action_token_id)
            
            # 将新生成的token添加到输入中，用于下一步生成
            input_ids = torch.cat([input_ids, action_token_id.unsqueeze(0)], dim=1)

        generated_segment = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_segment, total_log_prob.item()


    def update(self):
        """PPO更新逻辑"""
        print("\n--- [UPDATE] Starting Policy Update ---")
        
        # --- 1. 数据准备 ---
        # 从rollouts中提取数据
        states, actions, rewards, old_log_probs = [], [], [], []
        for rollout in self.rollouts:
            for step in rollout:
                states.append(step['state'])
                actions.append(step['generated_segment'])
                rewards.append(step['reward'])
                old_log_probs.append(step['log_prob'])

        # 转换为tensor
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(Config.DEVICE)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32).to(Config.DEVICE)

        # --- 2. 计算优势 ---
        with torch.no_grad():
            # Critic对每个高层状态进行价值评估
            state_inputs = self.tokenizer(states, return_tensors='pt', padding=True, truncation=True).to(Config.DEVICE)
            _, state_values = self.policy_old(state_inputs['input_ids'], state_inputs['attention_mask'])
            state_values = state_values.squeeze()
        
        # 优势 = 实际回报 - 价值评估
        advantages = rewards_tensor - state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # 标准化
        
        print(f"  Advantage Mean: {advantages.mean().item():.4f}")
        
        # --- 3. 多轮次PPO更新 ---
        for i in range(Config.K_EPOCHS):
            # a. 重新评估旧动作在新策略下的log_prob和价值
            #    我们将状态和动作拼接起来，评估生成动作的概率
            new_log_probs_list = []
            new_values_list = []
            for state, action in zip(states, actions):
                full_text = state + action
                input_ids = self.tokenizer.encode(full_text, return_tensors='pt').to(Config.DEVICE)
                attention_mask = torch.ones_like(input_ids)
                
                # 获取新策略的logits和value
                logits, value = self.policy(input_ids, attention_mask)
                
                # 计算生成action部分的log_prob
                action_ids = self.tokenizer.encode(action, add_special_tokens=False)
                state_len = len(self.tokenizer.encode(state))
                
                log_probs = F.log_softmax(logits, dim=-1)
                
                # 从logits中挑选出对应action_ids的log_prob
                # target_log_probs = log_probs[0, state_len-1:-1].gather(1, torch.tensor(action_ids).unsqueeze(1).to(Config.DEVICE)).squeeze()
                # 简化计算：我们只关心总的对数概率和价值
                # 在真实实现中，log_prob的计算会更精细
                
                new_log_probs_list.append(log_probs.mean()) # 简化版：用均值代替
                new_values_list.append(value.squeeze())

            new_log_probs = torch.stack(new_log_probs_list)
            new_values = torch.stack(new_values_list)
            
            # b. 计算重要性采样比率
            ratios = torch.exp(new_log_probs - old_log_probs_tensor)
            
            # c. PPO损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - Config.EPS_CLIP, 1 + Config.EPS_CLIP) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(new_values, rewards_tensor)
            
            total_loss = actor_loss + 0.5 * critic_loss
            
            # d. 优化
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            if i == 0:
                print(f"  Initial Loss: Total={total_loss.item():.4f}, Actor={actor_loss.item():.4f}, Critic={critic_loss.item():.4f}")

        print(f"  Final Loss:   Total={total_loss.item():.4f}, Actor={actor_loss.item():.4f}, Critic={critic_loss.item():.4f}")

        # --- 4. 更新旧策略 ---
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.rollouts.clear()
        print("--- [UPDATE] Update Complete ---\n")


# --- 4. 主训练循环 ---
def main():
    print(f"Running on device: {Config.DEVICE}")
    
    env = MockEnv()
    reward_model = RewardModel(env.target_text)
    agent = LLM_SHARSA_Agent()
    
    for episode in range(1, Config.MAX_EPISODES + 1):
        prompt = env.get_prompt()
        plan = env.get_plan()
        
        print(f"--- Episode {episode} ---")
        print(f"Prompt: {prompt}")
        print(f"Plan: {plan}")
        
        high_level_trajectory = []
        current_text = prompt
        full_generated_text = ""
        
        # --- 数据生成阶段 (Rollout) ---
        for subgoal in plan:
            # 低层执行
            generated_segment, log_prob = agent.generate_segment(current_text, subgoal)
            
            high_level_trajectory.append({
                'state': current_text,
                'subgoal': subgoal,
                'generated_segment': generated_segment,
                'log_prob': log_prob
            })
            
            current_text += generated_segment
            full_generated_text += generated_segment
            print(f"  Subgoal: '{subgoal}' -> Generated: '{generated_segment.strip()}'")

        # --- 奖励分配 ---
        final_reward = reward_model.evaluate(full_generated_text)
        print(f"Full Generated Text: '{full_generated_text.strip()}'")
        print(f"Final Reward: {final_reward:.4f}")
        
        for step in high_level_trajectory:
            step['reward'] = final_reward
        
        agent.rollouts.append(high_level_trajectory)
        
        # --- 学习与更新阶段 ---
        if episode % Config.UPDATE_AFTER_EPISODES == 0:
            agent.update()

if __name__ == "__main__":
    main()