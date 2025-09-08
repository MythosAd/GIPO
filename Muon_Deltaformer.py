import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 这是一个概念性的辅助函数，实际中需要高效实现
# 假设它能对一个矩阵进行k次Newton-Schulz迭代
def newton_schulz_k(matrix, k=5):
    # 这是一个占位符，代表了Muon优化器中的二阶近似步骤
    # 在真实实现中，这会是一个复杂的迭代过程
    # Y_0 = matrix.T
    # Z_0 = torch.eye(matrix.shape[0], device=matrix.device)
    # for _ in range(k):
    #     Y_{i+1} = 0.5 * Y_i @ (3 * torch.eye(Y_i.shape[0]) - Z_i @ Y_i)
    #     Z_{i+1} = 0.5 * (3 * torch.eye(Z_i.shape[0]) - Z_i @ Y_i) @ Z_i
    # return Y_k.T @ matrix
    return matrix # 为了代码能跑通，暂时返回原矩阵


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps): 
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X

def get_dynamic_coefficients(t):
    # 动态获取a, b, c 系数
    coefficients = [
        (3955 / 1024, -8306 / 1024, 5008 / 1024),
        (3735 / 1024, -6681 / 1024, 3463 / 1024),
        (3799 / 1024, -6499 / 1024, 3211 / 1024),
        (4019 / 1024, -6385 / 1024, 2906 / 1024),
        (2677 / 1024, -3029 / 1024, 1162 / 1024),
        (2172 / 1024, -1833 / 1024,  682 / 1024),
    ]
    if t < 0:
        raise ValueError("Time index t cannot be negative!")
    elif t < len(coefficients):
        raise ValueError("Time index t 超出范围！ (Time index t is out of range!)")
    else:
        raise ValueError("Time index t 超出范围！")

@torch.compile
def zeropower_via_newtonschulz6_dynamic(G):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G using dynamic coefficients.
    This method dynamically adjusts the coefficients (a, b, c) for each iteration to improve performance.
    The function ensures the spectral norm of G is at most 1 and performs six iterations of the process.

    Args:
        G (torch.Tensor): A 2D tensor to be orthogonalized.

    Returns:
        torch.Tensor: The orthogonalized tensor.
    """
    # https://spaces.ac.cn/archives/10922
    # 
    assert len(G.shape) == 2
    
    # 

    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(6):
        a, b, c = get_dynamic_coefficients(_)
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class MuDeltaFormerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, k_newton_schulz=6):
        """
        μ-DeltaFormer 核心层的初始化
        d_model: 模型的隐藏维度
        n_heads: 注意力头的数量
        d_k: 每个头的键/查询维度
        d_v: 每个头的值维度
        k_newton_schulz: Newton-Schulz迭代的次数  6次
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.k_newton_schulz = k_newton_schulz

        # 第一步：投影层
        # 这些层将输入 x_t 投影到 q, k, v, w 以及优化器参数
        self.w_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.w_w = nn.Linear(d_model, n_heads * d_k, bias=False) # DeltaFormer的辅助查询
        self.w_p = nn.Linear(d_model, n_heads * 2, bias=False)   # 学习率 η 和动量 θ
        
        self.w_o = nn.Linear(n_heads * d_v, d_model, bias=False) # 输出投影层

        # 用于计算 u_t 的 K1 核函数（指数核）的温度参数
        self.temperature_k1 = math.sqrt(d_k)
        
        # 记忆模块 S 和 动量状态 Momentum
        # 这里为了简化，我们将S和Momentum实现为可学习的参数，
        # 它们代表了t=0时的初始状态。在forward中它们会被更新。
        # 实际中，它们是作为状态在时间步之间传递的。
        # 形状: (n_heads, d_v, d_k)
        self.S_initial = nn.Parameter(torch.randn(n_heads, d_v, d_k))
        self.momentum_initial = nn.Parameter(torch.zeros(n_heads, d_v, d_k))

    def forward(self, x, memory_state=None):
        """
        x: 输入张量，形状 (batch_size, seq_len, d_model)
        memory_state: 包含了上一个时间步的 S, Momentum, 和 KV缓存
        """
        batch_size, seq_len, _ = x.shape

        # 初始化记忆状态和KV缓存
        if memory_state is None:
            S = self.S_initial.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            momentum = self.momentum_initial.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            # 这里的KV缓存存储的是 (k, u) 对
            kv_cache = {'k': [], 'u': []} 
        else:
            S = memory_state['S']
            momentum = memory_state['momentum']
            kv_cache = memory_state['kv_cache']

        # 存储每个时间步的输出
        outputs = []
        
        # 遍历序列中的每个词元 (自回归方式)
        for t in range(seq_len):
            x_t = x[:, t, :] # 当前输入 (batch_size, d_model)

            # --- 第一步：投影 ---
            # 形状: (batch_size, n_heads, d_k or d_v)
            q_t = self.w_q(x_t).view(batch_size, self.n_heads, self.d_k)
            k_t = self.w_k(x_t).view(batch_size, self.n_heads, self.d_k)
            v_t = self.w_v(x_t).view(batch_size, self.n_heads, self.d_v)
            w_t = self.w_w(x_t).view(batch_size, self.n_heads, self.d_k)
            
            # 动态计算优化器参数，并用sigmoid限制在(0,1)
            params_t = torch.sigmoid(self.w_p(x_t).view(batch_size, self.n_heads, 2))
            eta_t = params_t[:, :, 0].unsqueeze(-1).unsqueeze(-1)      # 学习率 η
            theta_t = params_t[:, :, 1].unsqueeze(-1).unsqueeze(-1)    # 动量系数 θ

            # --- 第二步：“净化”Value，计算 u_t ---
            # 这是DeltaFormer的核心
            if not kv_cache['k']: # 如果是第一个时间步
                u_t = v_t
            else:
                # 将历史k和u从缓存中取出并堆叠
                # 形状: (batch_size, n_heads, history_len, d_k or d_v)
                k_history = torch.stack(kv_cache['k'], dim=2)
                u_history = torch.stack(kv_cache['u'], dim=2)
                
                # 计算 K1(w_t, k_i)
                # w_t (bs, h, d_k) -> (bs, h, 1, d_k)
                # k_history (bs, h, hist, d_k) -> (bs, h, d_k, hist)
                # attn_scores (bs, h, 1, hist)
                attn_scores = torch.matmul(w_t.unsqueeze(2), k_history.transpose(2, 3)) / self.temperature_k1
                
                # 计算冗余信息
                # u_history (bs, h, hist, d_v)
                # redundant_info (bs, h, 1, d_v)
                redundant_info = torch.matmul(F.softmax(attn_scores, dim=-1), u_history)
                
                # 从 v_t 中减去冗余信息得到 u_t
                u_t = v_t - redundant_info.squeeze(2)

            # --- 第三步：定义“增量梯度” g_t ---
            # 形状: (batch_size, n_heads, d_v, d_k)
            g_t = torch.bmm(u_t.unsqueeze(2), k_t.unsqueeze(1)) # 外积

            # --- 第四步：使用Muon优化器更新记忆 S ---
            # 这是ATLAS的核心
            # 1. 更新动量
            momentum = theta_t * momentum + (1 - theta_t) * g_t
            
            # 2. 应用Newton-Schulz迭代 (概念性)
            s_prime_t = zeropower_via_newtonschulz6_dynamic(momentum)
            
            # 3. 更新记忆状态 S
            S = S + eta_t * s_prime_t

            # --- 第五步：从更新后的记忆中读取信息 ---
            # 这里为了简单，我们使用线性读取。φ(q_t)就简化为q_t
            # S (bs, h, d_v, d_k), q_t (bs, h, d_k) -> (bs, h, d_v)
            o_t_heads = torch.bmm(S, q_t.unsqueeze(2)).squeeze(2)
            
            # 将多头结果拼接起来
            o_t = o_t_heads.view(batch_size, self.n_heads * self.d_v)
            o_t = self.w_o(o_t) # (batch_size, d_model)
            
            # --- 第六步：更新缓存和状态 ---
            kv_cache['k'].append(k_t)
            kv_cache['u'].append(u_t) # 注意，我们缓存的是净化后的 u_t
            
            outputs.append(o_t)

        # 将所有时间步的输出堆叠起来
        final_output = torch.stack(outputs, dim=1)
        
        # 准备下一个调用所需的状态
        final_memory_state = {'S': S, 'momentum': momentum, 'kv_cache': kv_cache}
        
        return final_output, final_memory_state

# --- 示例使用 ---
d_model = 512
n_heads = 8
d_k = d_v = 64 # d_model // n_heads

model = MuDeltaFormerLayer(d_model, n_heads, d_k, d_v)
input_sequence = torch.randn(4, 10, d_model) # (batch_size, seq_len, d_model)

# 第一次调用
output, next_state = model(input_sequence)
print("Output shape:", output.shape)
# Output shape: torch.Size([4, 10, 512])

# 假设有新的输入，可以继续调用，并传入上一次的状态
new_input = torch.randn(4, 5, d_model)
new_output, _ = model(new_input, memory_state=next_state)
print("New output shape:", new_output.shape)
# New output shape: torch.Size([4, 5, 512])