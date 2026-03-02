# Day 5：面试冲刺——高频考点精讲 · 手写代码实战 · 知识体系总结

**学习目标**：经过前四天的系统学习，今天进入**面试冲刺阶段**。我们将系统梳理大模型面试的三大类高频问题（架构类、训练类、工程类），手写核心代码（自注意力、Transformer块、LoRA），并建立完整的知识体系图谱。

学完今天的内容，你将能够：自信应对大模型相关的技术面试，手写关键算法代码，并对整个大模型技术栈形成系统性认知。

---

## 第一部分：架构类高频面试题精讲

### 1.1 Transformer 相比 RNN 的优势是什么？

**标准答案框架**：

| 维度 | RNN | Transformer |
|:----|:----|:-----------|
| **并行性** | 必须串行，$t$ 时刻依赖 $t-1$ 的输出 | 完全并行，所有位置同时计算 |
| **长距离依赖** | 链式传递，容易梯度消失/爆炸 | 直接 attention，任意两点一步直达 |
| **训练效率** | 序列长度 $N$，时间复杂度 $O(N)$ 但无法并行 | $O(N^2)$ 但可并行，GPU 利用率高 |
| **可扩展性** | 难以扩展到大规模 | 可扩展到千亿参数 |

**加分回答**：

> "RNN 的本质问题是**信息必须沿时间步逐个传递**，这导致：1）无法并行训练；2）长序列信息在传递过程中会衰减或爆炸。Transformer 用 attention 机制让每个位置**直接看到所有其他位置**，彻底解决了这两个问题。代价是 $O(N^2)$ 的计算复杂度，但这可以通过 Flash Attention 等技术优化。"

---

### 1.2 为什么 Attention 要除以 $\sqrt{d_k}$？

**标准答案**：

防止点积结果过大导致 softmax 梯度消失。

**详细解释**：

假设 $Q$ 和 $K$ 的每个元素都是均值为 0、方差为 1 的独立随机变量：

$$
Q \cdot K = \sum_{i=1}^{d_k} q_i k_i
$$

根据方差的性质：$\text{Var}(Q \cdot K) = d_k$

当 $d_k = 64$ 时，点积的标准差是 $\sqrt{64} = 8$，这意味着点积值可能达到几十甚至上百。

**Softmax 的问题**：

```python
# 假设 scores = [10, 50, 30]
softmax([10, 50, 30]) ≈ [0.0000, 1.0000, 0.0000]  # 梯度几乎为0！

# 除以 sqrt(d_k) 后
scores / 8 = [1.25, 6.25, 3.75]
softmax([1.25, 6.25, 3.75]) ≈ [0.006, 0.920, 0.074]  # 梯度正常
```

---

### 1.3 多头注意力相比单头的优势？

**标准答案**：

多头注意力让模型能够**同时关注不同子空间的不同类型信息**。

**直观理解**：

```
单头：只有一种"看法"
  - 可能只学会了关注语法关系

多头（8头）：8 种不同"看法"
  - Head 1: 关注主谓关系
  - Head 2: 关注时态信息
  - Head 3: 关注指代关系
  - Head 4: 关注位置邻近
  - ...
```

**数学表达**：

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
$$

每个 head 有独立的 $W^Q_i, W^K_i, W^V_i$，学习不同的投影空间。

---

### 1.4 解释 MHA、MQA、GQA 的区别

这是**高频工程面试题**，考察你对现代 LLM 优化的理解。

| 方法 | Q 头数 | KV 头数 | KV Cache 大小 | 效果 |
|:----|:------|:-------|:-------------|:----|
| **MHA** | 8 | 8 | 基准 | 最好 |
| **MQA** | 8 | 1 | 1/8 | 略差 |
| **GQA** | 8 | 2 | 1/4 | 接近 MHA |

**代码层面的区别**：

```python
# MHA: 每个 Q 头有独立的 KV
W_k = nn.Linear(hidden, 8 * head_dim)  # 8 组 KV

# GQA: 4 个 Q 头共享 1 组 KV
W_k = nn.Linear(hidden, 2 * head_dim)  # 只有 2 组 KV
# 计算时用 repeat_kv 复制成 8 组

# MQA: 所有 Q 头共享 1 组 KV
W_k = nn.Linear(hidden, 1 * head_dim)  # 只有 1 组 KV
```

**为什么 GQA 是主流？**

> "GQA 是 MHA 和 MQA 的折中。MQA 太激进，所有 Q 头共享一组 KV 会损失表达能力；MHA 的 KV Cache 太大。GQA 用少量 KV 组（如 2 或 4），既大幅减少显存，又保持接近 MHA 的效果。LLaMA 2/3、MiniMind 都采用 GQA。"

---

### 1.5 RoPE 相比绝对位置编码的优势？

**标准答案**：

1. **相对位置信息**：虽然是加在输入上的"绝对"编码，但经过 $Q \cdot K^T$ 后自然产生相对位置效果
2. **长度外推**：训练时最长 2048，推理时可外推到 32K+
3. **计算高效**：只需要 $O(d)$ 的额外计算

**原理简述**：

RoPE 把位置 $m$ 编码为一个旋转角度，让 $Q_m$ 和 $K_n$ 的点积结果只依赖于**相对位置 $m-n$**：

$$
\langle \text{RoPE}(q, m), \text{RoPE}(k, n) \rangle = f(q, k, m-n)
$$

这是因为旋转矩阵相乘时，角度会相减。

---

## 第二部分：训练类高频面试题精讲

### 2.1 预训练和微调的区别？

| 维度 | 预训练 (Pretrain) | 微调 (Fine-tune/SFT) |
|:----|:-----------------|:-------------------|
| **目标** | 学习语言通用知识 | 学习特定任务/对话能力 |
| **数据** | 海量无标注文本 | 少量高质量标注数据 |
| **学习率** | 较大 (1e-4 ~ 1e-3) | 较小 (1e-5 ~ 1e-4) |
| **训练量** | 数万亿 token | 数百万 ~ 数十亿 token |
| **损失函数** | Next Token Prediction | 同左，但只在回答部分计算 loss |

**加分回答**：

> "预训练让模型学会'什么是语言'，微调让模型学会'怎么对话'。预训练的数据量大但质量参差，所以用较大学习率快速学习；微调数据量小但质量高，用小学习率精细调整，避免遗忘预训练知识（灾难性遗忘）。"

---

### 2.2 LoRA 的原理是什么？为什么有效？

**核心原理**：

大模型微调时，参数变化量 $\Delta W$ 是**低秩的**，可以用两个小矩阵近似：

$$
W' = W_0 + \Delta W = W_0 + BA
$$

其中 $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d, k)$

**参数量对比**：

```
原始: d × k = 512 × 512 = 262,144
LoRA: r × (d + k) = 8 × 1024 = 8,192  ← 节省 97%！
```

**为什么 B 初始化为 0？**

> "如果 B 随机初始化，训练开始时 LoRA 分支会输出随机噪声，破坏预训练权重的效果。B 全零初始化确保 $\Delta W = BA = 0$，训练起点和原模型完全等价。"

**MiniMind LoRA 实现**：

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)   # 降维
        self.B = nn.Linear(rank, out_features, bias=False)  # 升维

        self.A.weight.data.normal_(mean=0.0, std=0.02)  # A: 高斯初始化
        self.B.weight.data.zero_()                       # B: 零初始化！

    def forward(self, x):
        return self.B(self.A(x))  # x → 压缩 → 还原
```

---

### 2.3 DPO 和 RLHF 的区别？

| 维度 | RLHF (PPO) | DPO |
|:----|:----------|:----|
| **模型数量** | 策略 + 参考 + 奖励模型 | 策略 + 参考模型 |
| **训练稳定性** | PPO 调参困难 | 稳定，本质是监督学习 |
| **数据格式** | 需要奖励分数 | 只需要 (chosen, rejected) 对 |
| **计算成本** | 高（三个模型） | 较低 |

**DPO 的核心洞察**：

> "DPO 证明了一个数学等价性：奖励模型可以被**隐式编码**在策略模型里。通过直接对比 chosen 和 rejected 在策略模型 vs 参考模型上的概率差，就能得到和 RLHF 相同的优化方向，无需显式训练奖励模型。"

**DPO Loss（简化版）**：

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
$$

---

### 2.4 训练时 Loss 不下降怎么办？

**排查清单**：

| 可能原因 | 检查方法 | 解决方案 |
|:--------|:--------|:--------|
| **学习率太大** | 看 loss 是否震荡 | 降低 10 倍 |
| **学习率太小** | 看 loss 几乎不动 | 提高 10 倍 |
| **数据有问题** | 打印几个 batch 看看 | 检查数据处理逻辑 |
| **梯度消失/爆炸** | 打印梯度范数 | 加梯度裁剪、检查初始化 |
| **label 计算错误** | 检查 shift 逻辑 | 确保 label 是 input 右移一位 |
| **模型太小** | 尝试更大模型 | 增加层数/宽度 |

**必会的调试代码**：

```python
# 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm():.4f}")

# 检查 loss 的各个分量
print(f"loss: {loss.item():.4f}")
print(f"logits range: [{logits.min():.2f}, {logits.max():.2f}]")
```

---

## 第三部分：工程类高频面试题精讲

### 3.1 KV Cache 的作用和代价？

**作用**：

避免自回归生成时重复计算前面位置的 K、V，把 $O(N^2)$ 计算降到 $O(N)$。

**代价**：

显存占用随序列长度线性增长：

$$
\text{KV Cache} = 2 \times L \times N \times d \times \text{batch} \times \text{bytes}
$$

**MiniMind 示例**（8层，512维，FP16）：
- 生成 4096 token：$2 \times 8 \times 4096 \times 512 \times 2 = 64\text{MB}$（单样本）

**优化方向**：GQA（减少 KV 头数）、Paged Attention（动态分配）、Sliding Window（只缓存最近 N 个）

---

### 3.2 Flash Attention 的加速原理？

**核心洞察**：GPU 计算快，但显存读写（IO）慢。

**标准 Attention 的问题**：

```
1. 算出 N×N 的 attention 矩阵 → 写入 HBM（慢）
2. 读出来做 softmax            → 读取 HBM（慢）
3. 结果写回去                  → 写入 HBM（慢）
4. 读出来乘 V                  → 读取 HBM（慢）
```

**Flash Attention 的做法**：

```
1. 把 Q、K、V 分成小块
2. 每个小块在 SRAM（高速缓存）里算完 attention
3. 用在线 softmax 合并各块结果
4. 直接输出，不存中间的 N×N 矩阵
```

**效果**：显存 $O(N^2) \to O(N)$，速度快 2-4 倍，结果**数学上完全等价**。

---

### 3.3 分布式训练的并行策略？

| 策略 | 切分方式 | 通信量 | 适用场景 |
|:----|:--------|:------|:--------|
| **数据并行 (DP/DDP)** | 数据切分，每卡完整模型 | 梯度 AllReduce | 最常用 |
| **模型并行 (MP)** | 模型按层切分 | 激活值前传 | 超大模型 |
| **张量并行 (TP)** | 单层内部切分 | 每层都要通信 | 超大模型 |
| **流水线并行 (PP)** | 按阶段切分 | 微批次流水 | 减少气泡 |
| **ZeRO/FSDP** | 切分优化器状态/梯度/参数 | 按需 AllGather | 极限省显存 |

**DDP 的核心代码**：

```python
# 初始化
dist.init_process_group(backend="nccl")
model = DistributedDataParallel(model, device_ids=[local_rank])

# 数据采样
sampler = DistributedSampler(dataset)
loader = DataLoader(dataset, sampler=sampler)

# 每个 epoch 设置 sampler
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # 确保每个 epoch 数据打乱方式不同
```

---

### 3.4 量化会损失精度吗？如何减少损失？

**会损失**，因为用离散整数近似连续浮点数必然有舍入误差。

**减少损失的方法**：

| 方法 | 原理 |
|:----|:----|
| **逐通道量化** | 每个通道独立计算 scale，比全层量化更精确 |
| **校准数据集** | 用少量真实数据校准 scale 因子 |
| **QAT** | 训练时模拟量化误差，让模型适应 |
| **混合精度** | 敏感层保持高精度，其他层量化 |
| **GPTQ/AWQ** | 先进的 4-bit 量化算法，精度损失极小 |

---

## 第四部分：手写代码实战

### 4.1 手写 Scaled Dot-Product Attention

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: (batch, heads, seq_len, head_dim)
        K: (batch, heads, seq_len, head_dim)
        V: (batch, heads, seq_len, head_dim)
        mask: (batch, 1, 1, seq_len) 或 (seq_len, seq_len)
    Returns:
        output: (batch, heads, seq_len, head_dim)
    """
    d_k = Q.size(-1)

    # 1. Q @ K^T，计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: (batch, heads, seq_len, seq_len)

    # 2. 应用 mask（如因果掩码）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 3. Softmax 归一化
    attention_weights = F.softmax(scores, dim=-1)

    # 4. 加权求和
    output = torch.matmul(attention_weights, V)

    return output


def causal_mask(seq_len):
    """生成因果掩码（下三角矩阵）"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # (seq_len, seq_len)


# 测试
batch, heads, seq_len, head_dim = 2, 8, 10, 64
Q = torch.randn(batch, heads, seq_len, head_dim)
K = torch.randn(batch, heads, seq_len, head_dim)
V = torch.randn(batch, heads, seq_len, head_dim)
mask = causal_mask(seq_len)

output = scaled_dot_product_attention(Q, K, V, mask)
print(f"Output shape: {output.shape}")  # (2, 8, 10, 64)
```

---

### 4.2 手写简化版 Transformer Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """RMS Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class MultiHeadAttention(nn.Module):
    """多头注意力（简化版，不含 GQA）"""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape

        # 投影
        Q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(output)


class FeedForward(nn.Module):
    """SwiGLU 前馈网络"""
    def __init__(self, hidden_size, intermediate_size=None):
        super().__init__()
        intermediate_size = intermediate_size or int(hidden_size * 8 / 3)

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        # SwiGLU: SiLU(gate) * up
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """完整的 Transformer Block (Pre-LN)"""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.feed_forward = FeedForward(hidden_size)
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)

    def forward(self, x, mask=None):
        # Pre-LN: Norm → Attention → Residual
        x = x + self.attention(self.norm1(x), mask)
        # Pre-LN: Norm → FFN → Residual
        x = x + self.feed_forward(self.norm2(x))
        return x


# 测试
batch, seq_len, hidden_size, num_heads = 2, 10, 512, 8
x = torch.randn(batch, seq_len, hidden_size)
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

block = TransformerBlock(hidden_size, num_heads)
output = block(x, mask)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")  # 相同
print(f"Params: {sum(p.numel() for p in block.parameters()) / 1e6:.2f}M")
```

---

### 4.3 手写 LoRA 模块

```python
import torch
import torch.nn as nn


class LoRA(nn.Module):
    """LoRA 低秩适配器"""
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # 缩放因子

        # A: 降维矩阵
        self.A = nn.Linear(in_features, rank, bias=False)
        # B: 升维矩阵
        self.B = nn.Linear(rank, out_features, bias=False)

        # 初始化
        nn.init.normal_(self.A.weight, mean=0, std=0.02)
        nn.init.zeros_(self.B.weight)  # B 初始化为 0！

    def forward(self, x):
        # ΔW = B @ A, 输出 = x @ ΔW^T * scaling
        return self.B(self.A(x)) * self.scaling


class LinearWithLoRA(nn.Module):
    """在原始 Linear 上挂载 LoRA"""
    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()
        self.original = original_linear
        self.lora = LoRA(
            original_linear.in_features,
            original_linear.out_features,
            rank=rank,
            alpha=alpha
        )
        # 冻结原始权重
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x):
        # 原始输出 + LoRA 输出
        return self.original(x) + self.lora(x)


def apply_lora_to_model(model, rank=8, alpha=16, target_modules=['q_proj', 'v_proj']):
    """给模型的指定层添加 LoRA"""
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # 替换为带 LoRA 的版本
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name, LinearWithLoRA(module, rank, alpha))

    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.2f}%)")

    return model


# 测试
original = nn.Linear(512, 512)
lora_linear = LinearWithLoRA(original, rank=8)

x = torch.randn(2, 10, 512)
output = lora_linear(x)
print(f"Output shape: {output.shape}")

# LoRA 参数量
lora_params = sum(p.numel() for p in lora_linear.lora.parameters())
original_params = sum(p.numel() for p in original.parameters())
print(f"Original params: {original_params}")
print(f"LoRA params: {lora_params} ({100*lora_params/original_params:.2f}%)")
```

---

## 第五部分：知识体系总结

### 5.1 大模型知识地图

```
大模型知识体系
├── 基础架构
│   ├── Transformer 核心
│   │   ├── Self-Attention（Q、K、V 机制）
│   │   ├── Multi-Head Attention（多头并行）
│   │   ├── Position Encoding（RoPE 旋转位置编码）
│   │   └── Feed-Forward Network（SwiGLU 门控）
│   │
│   ├── 现代优化
│   │   ├── RMSNorm（替代 LayerNorm）
│   │   ├── GQA（减少 KV 头数）
│   │   ├── Pre-LN（先归一化再计算）
│   │   └── MoE（混合专家系统）
│   │
│   └── 模型结构
│       ├── Decoder-Only（GPT/LLaMA/MiniMind）
│       ├── Encoder-Only（BERT）
│       └── Encoder-Decoder（T5/BART）
│
├── 训练流程
│   ├── 预训练（Pretrain）
│   │   ├── 目标：Next Token Prediction
│   │   ├── 数据：海量无标注文本
│   │   └── 技巧：学习率调度、梯度裁剪、混合精度
│   │
│   ├── 有监督微调（SFT）
│   │   ├── 目标：学会对话格式
│   │   ├── 数据：指令-回答对
│   │   └── 技巧：只在回答部分计算 loss
│   │
│   ├── 对齐训练
│   │   ├── RLHF（PPO + 奖励模型）
│   │   └── DPO（直接偏好优化，无需奖励模型）
│   │
│   └── 参数高效微调
│       ├── LoRA（低秩适配）
│       ├── QLoRA（量化 + LoRA）
│       └── Adapter（适配器层）
│
├── 推理优化
│   ├── 显存优化
│   │   ├── KV Cache（缓存 K、V 避免重复计算）
│   │   ├── Flash Attention（IO 感知分块计算）
│   │   └── Paged Attention（动态显存管理）
│   │
│   ├── 量化
│   │   ├── INT8/INT4 量化
│   │   ├── GPTQ/AWQ（高精度 4-bit）
│   │   └── GGUF/GGML（本地部署格式）
│   │
│   └── 加速框架
│       ├── vLLM（高吞吐推理）
│       ├── TensorRT-LLM（NVIDIA 优化）
│       └── llama.cpp（CPU 推理）
│
└── 分布式训练
    ├── 数据并行（DDP）
    ├── 模型并行（MP/TP/PP）
    └── ZeRO/FSDP（切分一切冗余）
```

### 5.2 面试速查表

**架构类**：

| 问题 | 关键词 |
|:----|:------|
| Transformer vs RNN | 并行、长距离依赖、可扩展 |
| 为什么除以 √d_k | 防止 softmax 梯度消失 |
| 多头注意力作用 | 多子空间、不同语义关系 |
| RoPE 优势 | 相对位置、长度外推 |
| GQA 原理 | Q 头多、KV 头少、折中方案 |

**训练类**：

| 问题 | 关键词 |
|:----|:------|
| 预训练 vs 微调 | 知识 vs 能力、数据量、学习率 |
| LoRA 原理 | 低秩、BA 分解、B 零初始化 |
| DPO vs RLHF | 无需奖励模型、更稳定 |
| Loss 不下降 | 学习率、数据、梯度、label |

**工程类**：

| 问题 | 关键词 |
|:----|:------|
| KV Cache | 避免重复、空间换时间 |
| Flash Attention | IO 优化、分块、在线 softmax |
| 分布式策略 | DP/MP/TP/PP/ZeRO |
| 量化损失 | 逐通道、校准、QAT |

---

## 🚀 最终检查清单

### 理论知识 ✓
- [ ] 能画出 Transformer 整体架构图
- [ ] 能解释 Self-Attention 的 Q、K、V 含义
- [ ] 能说清 RoPE 的优势
- [ ] 能对比 MHA/MQA/GQA
- [ ] 能解释预训练和微调的区别
- [ ] 能说清 LoRA 为什么 B 初始化为 0
- [ ] 能对比 DPO 和 RLHF
- [ ] 能解释 KV Cache 的原理和代价
- [ ] 能说清 Flash Attention 为什么快

### 手写代码 ✓
- [ ] 能手写 Scaled Dot-Product Attention
- [ ] 能手写简化版 Transformer Block
- [ ] 能手写 LoRA 模块
- [ ] 能手写 RMSNorm
- [ ] 能手写因果掩码

### 工程经验 ✓
- [ ] 跑过 MiniMind 的预训练和 SFT
- [ ] 理解分布式训练的 DDP 配置
- [ ] 知道如何排查 Loss 不下降的问题
- [ ] 了解主流的推理优化框架

---

## 后续学习建议

**必读论文**：
1. *Attention Is All You Need*（Transformer 原始论文）
2. *LLaMA: Open and Efficient Foundation Language Models*
3. *LoRA: Low-Rank Adaptation of Large Language Models*
4. *Direct Preference Optimization*
5. *Flash Attention: Fast and Memory-Efficient Exact Attention*

**实践项目**：
1. 在 MiniMind 上实现自己的微调任务
2. 尝试用 LoRA 微调一个特定领域的模型
3. 部署模型到 vLLM 或 llama.cpp
4. 参与开源项目贡献

**持续关注**：
- Hugging Face 博客
- 各大厂的技术博客（Meta AI、Google Research、OpenAI）
- arXiv 每日更新

---

> 🎉 **恭喜完成 5 天速成学习！**
>
> 你已经系统掌握了大模型的核心知识体系：从 Transformer 架构到训练流程，从推理优化到分布式训练。现在，带着这些知识去面试、去实践、去创造吧！
>
> 记住：**理解原理 + 动手实践 + 持续学习 = 真正掌握**
