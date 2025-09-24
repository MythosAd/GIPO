---
title: "GIPO: Group Information Policy Optimization for large language model"
author: "杨诚操"
date: "2025-09-04"
categories: [ agent, AGI]
---

## 基于贝叶斯代理的GIPO框架推导

### 1. 起点：信息瓶颈（IB）原理

我们的理论起点是信息瓶颈（Information Bottleneck, IB）原理。该原理旨在寻找一个信息瓶颈变量 $r$（在我们的场景中是LLM生成的推理链），它在压缩输入 $q$（问题）的同时，最大化地保留与目标 $a$（答案）相关的信息。

这被形式化为以下优化目标：
$$
\min_{\pi(r|q)} \mathcal{L}_{IB}(\pi) = I(q; r) - \beta I(r; a)
$$
其中，$(\pi(r|q)$ 是模型生成推理链的策略，$\beta > 0$ 是权衡参数（较大 $\beta$ 鼓励更多压缩）。互信息 $I(\cdot;\cdot)$ 基于联合分布 $p(q,r,a)$，其中 $r \sim \pi(r|q)$，$a \sim p(a|q)$。我们的任务是找到一个最优策略 $\pi^*$ 来最大化此目标。

### 2. 推导步骤：从互信息到可操作的代理目标

我们将对 $\mathcal{L}_{IB}$ 的两个组成部分——信息项 $$I(r;a)$$ 和压缩项 $$I(q;r)$$——分别进行分解和近似。

#### **第一步：处理压缩项 $I(q;r)$**

根据互信息的定义，我们有：
$$
I(q; r) = H(r) - H(r|q)
$$
其中 $H(r) = -\mathbb{E}_{p(r)}[\log p(r)]$ 是推理链 $r$ 的先验熵，$H(r|q) = \mathbb{E}_{p(q)} [-\mathbb{E}_{\pi(r|q)}[\log \pi(r|q)]]$ 是在给定问题 $q$ 的条件下 $r$ 的条件熵。

我们引入一个关键的实践假设：

**假设 1**: 在强化学习微调的有限范围内，模型生成文本的整体边缘分布 $p(r) = \mathbb{E}_{p(q)} \pi(r|q)$ 相对稳定，从而 $H(r)$ 可以被视为一个与策略 $\pi$ 的优化无关的近似常数。

这一项的目标是最大化条件熵 $H(r|q)$，它鼓励模型在给定问题时生成更多样化的推理路径，从而促进探索并提高泛化能力。
这是一个变分近似（类似于变分自编码器中的边缘KL近似），但需注意其局限性：如果优化显著改变 $\pi(r|q)$（例如，从随机到高度确定性），$H(r)$ 可能发生变化，导致轻微偏差。因此，最小化 $I(q;r)$ 的目标在优化过程中近似等价于：
$$
\min_{\pi} I(q;r) \approx \min_{\pi} [H(r) - H(r|q)] \iff \max_{\pi} H(r|q)
$$
这一项鼓励最大化条件熵 $H(r|q)$，促进模型在给定问题时生成多样化的推理路径，从而提升探索和泛化能力。在高维序列空间中，$H(r|q)$ 通常通过采样或PPO-style的熵计算来近似。

#### **第二步：处理信息项 $I(r;a)$**

这是本次推导的核心所在。我们同样从互信息的定义出发：
$$
I(r; a) = H(a) - H(a|r)
$$
其中 $H(a) = -\mathbb{E}_{p(a)}[\log p(a)]$ 是答案的边缘熵，$H(a|r) = \mathbb{E}_{p(r)} [-\mathbb{E}_{p(a|r)}[\log p(a|r)]]$ 是条件熵。

$H(a)$ 是由数据集本身的分布决定的，与我们优化的模型策略 $\pi$ 无关，因此它也是一个常数。最大化 $I(r;a)$ 等价于最小化 $H(a|r)$，其直观含义是：好的推理链 $r$ 应显著减少关于答案 $a$ 的不确定性。

#### **第三步：构建 $H(a|r)$ 的可计算代理**

直接最小化 $H(a|r)$ 仍然是困难的，因为它的定义涉及真实的、未知的后验分布 $p(a|r)$：
$$
H(a|r) = -\mathbb{E}_{p(r,a)}[\log p(a|r)]
$$

* 在监督学习中，我们只知道 **数据分布** $p(a|q)$，即给定问题 $q$ 时答案的分布（通常是确定的正确答案）。
* 但推理链 $r$ 是由模型策略 $\pi(r|q)$ 生成的，不是原始数据的输入特征。数据集中并没有直接标注 “给定推理链 $r$，答案的条件分布”。
* 要得到 $p(a|r)$，必须通过 **贝叶斯反演**：
$$
p(a|r) = \frac{p(r|a)p(a)}{p(r)}.
$$

其中 $p(r|a)$、$p(r)$ 在实践中不可直接获取，也难以估计。

换句话说，**我们没有直接观测 $p(a|r)$，只能观测 $p(a|q)$**。因此 $H(a|r)$ 不能被精确计算。

为了构建一个可操作的代理目标，我们使用模型 $\pi_\theta$ 自身来近似 $p(a|r)$。并引入交叉熵作为上界（变分上界）：具体来说，我们假设模型在生成推理链 $r$ 之后，具备预测最终答案 $a$ 的能力，这个预测分布我们记为 $\pi_\theta(a|r)$。
$$
H(a|r) \leq - \mathbb{E}_{r,a}[\log \pi_\theta(a|r)].
$$

具体而言，我们用交叉熵 (CE) 作为 $H(a|r)$ 的变分上界（因为 $CE = H + KL \geq H$，且 $KL \geq 0）$ ： 

$$
\begin{align*}
CE(p, \pi_\theta)
& \equiv -\sum_a p(a|r) \log \pi_\theta(a|r) \\
& = -\sum_a p(a|r) \log \pi_\theta(a|r) + \sum_a p(a|r) \log p(a|r) - \sum_a p(a|r) \log p(a|r) \\
& = \left( -\sum_a p(a|r) \log p(a|r) \right) + \left( \sum_a p(a|r) \log p(a|r) - \sum_a p(a|r) \log \pi_\theta(a|r) \right) \\
& = H(a|r) + \sum_a p(a|r) \left( \log p(a|r) - \log \pi_\theta(a|r) \right) \\
& = H(a|r) + \sum_a p(a|r) \log \frac{p(a|r)}{\pi_\theta(a|r)} \\
& = H(a|r) + D_{KL}(p(a|r) || \pi_\theta(a|r)) \\
& \ge H(a|r)
\end{align*}
$$
因此可以得到
$$ 
H(a|r) \leq -\mathbb{E}_{r \sim \pi_\theta(\cdot|q), a \sim p(\cdot|q)} [\log \pi_\theta(a|r)] 
$$
因此，最小化 $H(a|r)$ 可以近似为最小化这个上界：
$$
\min_{\pi_\theta} H(a|r) \approx \min_{\pi_\theta} \left( -\mathbb{E}_{r \sim \pi_\theta(\cdot|q), a \sim p(\cdot|q)}[\log \pi_\theta(a|r)] \right)
$$
这个代理目标直观：它要求生成的 $r$ 让模型以高置信度预测正确答案。该近似假设 $\pi_\theta(a|r) \approx p(a|r)$，KL散度在迭代优化中趋于零；否则，可能引入偏差，但这在实践中通过自我一致性训练缓解。注意，期望采样假设 $a$ 给定 $q$ 独立于 $r$，但因果链 $q \to r \to a$ 通过贝叶斯反演隐式处理。

#### **第四步：组合成最终的代理目标**

现在，我们将处理过的压缩项和信息项代回到原始的IB目标中。
$$
\min_{\pi_\theta} \mathcal{L}_{GIPO}(\pi_\theta) = \min_{\pi_\theta} [I(q;r) - \beta I(r;a)]
$$
忽略优化过程中的常数项 $H(r)$ 和 $H(a)$，我们得到：
$$
\min_{\pi_\theta} [-H(r|q) - \beta(-H(a|r))] = \min_{\pi_\theta} [-H(r|q) + \beta H(a|r)]
$$
最后，将 $H(a|r)$ 替换为其可计算的交叉熵代理，我们得到最终的、基于贝叶斯代理的GIPO优化目标：
$$ 
\mathcal{J}_{alt}(\pi_\theta) = \max_{\pi_\theta} \left( \mathbb{E}_{r \sim \pi_\theta(\cdot|q), a \sim p(\cdot|q)} [\log \pi_\theta(a|r)] + \beta H(r|q) \right) 
$$

### 3. 代理目标的解读与实现

这个新推导出的代理目标 $\mathcal{J}_{alt}(\pi_\theta)$ 可以分解为两个相互制衡的部分：

1.  **最小化 $-H(r|q)$**: 等价于**最大化条件熵 $H(r|q)$**。
    *   **作用**: 作为熵正则化项，鼓励模型探索不同的推理路径，防止策略过早收敛，提升泛化性。
    *   **实现**: 在RL中，这对应于在PPO等算法的目标函数中加入熵奖励项。

2.  **最小化 $-\beta \cdot \mathbb{E}[\log \pi_\theta(a|r)]$**: 等价于**最大化答案预测的对数似然**。
    *   **作用**: 强制要求生成的推理链 $r$ 必须包含足够充分且清晰的信息来唯一地确定答案。它为“好的推理”提供了一个非常具体和可量化的标准。
    *   **实现**: 在RL框架中，$\log \pi_\theta(a|r)$ 可以被实现为一种**内在奖励（Intrinsic Reward）**。在每次模型生成一个完整的推理链 $r_i$ 后，我们计算它预测出正确答案 $a_i$ 的对数概率。这个值可以作为一个额外的奖励信号，与环境提供的外部奖励（例如，答案对错的0/1奖励）结合，共同指导策略的更新。这种奖励塑形（Reward Shaping）机制能够为模型提供更密集的学习信号，尤其是在外部奖励稀疏的情况下。

### 4. 结论

通过上述推导，我们从IB第一性原理出发，构建了一个理论上更紧凑、因果关系更明确的代理目标。与原论文依赖启发式近似（用优势函数 $A_t$ 替代信息重要性）的方法不同，本路径直接将“信息性” $I(r;a)$ 与一个可计算的、有明确意义的机器学习目标（最大化答案预测置信度）联系起来。

尽管该方法需要额外的计算开销（用于计算 $\pi_\theta(a|r)$），但它为IB理论在LLM推理优化中的应用提供了一条更严谨、更具可解释性的实现路径。


### 关于熵项 $-H(r|q)$ 是否需要衰减（decay）？


*   **推理初期**: 模型需要探索不同的可能性和解题路径。此时，较高的熵是有益的，可以防止模型过早地锁定在一个次优的思路上。
*   **推理中期**: 随着关键步骤的确定，推理路径应该逐渐收敛。不确定性应当开始降低，逻辑链条变得更加清晰和确定。
*   **推理末期/接近答案**: 此时，几乎所有的不确定性都应该被消除。推理应该以高度确定的方式指向唯一的答案。熵应该非常低。


我们可以引入一个随时间步 $t$ 变化的衰减因子 $\gamma(t)$ 来修正这一项。

**修正后的代理目标**:
$$
\mathcal{J}_{alt-decay}(\pi_\theta) = \min_{\pi_\theta} \left( -\sum_{t=1}^T \gamma(t) H(o_t|o_{<t},q) - \beta \cdot \mathbb{E}[\log \pi_\theta(a_{gold}|r)] \right)
$$
其中，$\gamma(t)$ 是一个非负的、随 $t$ 单调递减的函数。例如：

*   **线性衰减**: $\gamma(t) = \max(0, 1 - \frac{t}{T_{max}})$，其中 $T_{max}$ 是一个预设的最大长度。
*   **指数衰减**: $\gamma(t) = \gamma_0^t$，其中 $0 < \gamma_0 < 1$ 是衰减率。
*   **分段函数**: 在推理的前 $k$ 步 $\gamma(t)=1$，之后 $\gamma(t)$ 开始衰减。

## Citation

```
@article{he2025nondeterminism,
  author = {ChengCao Yang},
  title = {GIPO: Group Information Policy Optimization for large language model},
  year = {2025},
  note = {https://mythosad.github.io/posts/GSIP/},
}
```
