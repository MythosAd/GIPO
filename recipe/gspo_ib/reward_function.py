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
# from . import gsm8k, math, prime_math, prime_code
import re
def compute_score(
    data_source,
    solution_str,
    ground_truth,
    solution_ids,  # 新增：传入原始的 token ID 列表
    tokenizer,     # 新增：传入 tokenizer 实例
    extra_info=None,
    # ... 其他参数
):
    """
    计算给定解决方案的分数，并在 token 层面切分 reasoning 和 answer。
    """
    if data_source == "openai/gsm8k":
        from verl.utils.reward_score import gsm8k
        base_score = gsm8k.compute_score(solution_str, ground_truth, method="strict")

        reasoning_token_len = 0
        answer_token_len = 0
        has_answer = False

        # 仍然在字符串层面寻找答案分隔符 "####"
        answer_match = re.search(r"####\s*(-?[\d\.,]+)", solution_str)
                
        # 获取响应部分的总 token 数量
        total_response_tokens = len(solution_ids)

        if answer_match:
            has_answer = True
            # 找到 "####" 在字符串中的起始位置
            answer_char_start_pos = answer_match.start()
            
            # --- 核心逻辑：将字符位置映射回 Token 位置 ---
            # 这是一个稳健的方法，通过逐个解码 token 来累加字符长度
            current_char_len = 0
            split_token_index = 0
            for i, token_id in enumerate(solution_ids):
                # 解码单个 token。注意：不要跳过特殊 token，但要清理空格
                token_str = tokenizer.decode([token_id])
                if current_char_len >= answer_char_start_pos:
                    break
                current_char_len += len(token_str)
                split_token_index = i + 1

            reasoning_token_len = split_token_index
            answer_token_len = total_response_tokens - reasoning_token_len
        else:
            # 如果没有找到答案，整个响应都算作 reasoning
            reasoning_token_len = total_response_tokens
            answer_token_len = 0
            has_answer = False
            
        # 返回 token 长度，而不是字符串
        return {
            "score": base_score,
            "reasoning_token_len": reasoning_token_len,
            "answer_token_len": answer_token_len,
            "has_answer": has_answer,
        }


__all__ = ["compute_score"]
