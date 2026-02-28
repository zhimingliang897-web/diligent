# Day 3：进阶炼丹术——RLHF · DPO · LoRA 与模型量化全解析

**学习目标**：在掌握了预训练与 SFT 的基础之上，今天我们将攻克让大模型从"能对话"到"说人话"的对齐技术——**RLHF 与 DPO**，并深入 MiniMind 源码手术台，拆解参数高效微调的核心武器 **LoRA**，最后了解让大模型在消费级设备上跑起来的**量化技术**。

学完今天的内容，你将能从代码层面理解：为什么 DPO 能绕过训练奖励模型，LoRA 的 A、B 两矩阵为何一个高斯初始化、一个零初始化，以及量化的精度代价从何而来。

---

## 第一部分：从 SFT 到对齐——为什么还需要 RLHF？

经过预训练和 SFT，模型已经能像模像样地回答问题。但它仍然可能：

- 一本正经地胡说八道（幻觉）
- 对有害请求照单全收
- 给出两个答案都"说得通"，但人类明显更喜欢其中一个

这些问题靠 SFT 数据很难根治，因为 **SFT 只能教模型"该说什么"，却很难教它"什么好、什么坏"**。这就是 RLHF 的出场时机。

### 1.1 RLHF 的核心动机：偏好排序成本远低于答案生成


| 标注方式                 | 难度                    | 成本 |
| :----------------------- | :---------------------- | :--- |
| 写一条高质量 SFT 答案    | 高（需要专业知识+时间） | 贵   |
| 在两个答案里选更好的那个 | 低（普通用户就能做）    | 便宜 |

RLHF 的精华就是把"人类更喜欢哪个"这种主观判断，转化成可以驱动梯度下降的数学信号。

### 1.2 RLHF 三步走战略

#### 第一步：SFT 热身（Supervised Fine-Tuning）

把一个预训练好的 Base 模型，用指令数据微调成能进行基础对话的助手。这一步我们昨天已经深入学过了。

#### 第二步：训练奖励模型（Reward Model, RM）

1. 给 SFT 模型同一个问题，让它生成多个不同的回答（A、B、C、D）；
2. 人类标注员给这些回答排序，比如 A > B > C = D；
3. 用这些排序数据训练一个**奖励模型**——它的任务只有一个：**给任意一条回答打出一个分数**，这个分数要和人类的偏好方向一致。

奖励模型通常是一个比主模型小的 Transformer，它的最后一层输出一个标量分数而不是词表维度的 logits。

#### 第三步：PPO 强化学习（Proximal Policy Optimization）

有了"打分裁判"（RM）之后，主模型开始和它博弈：

- 模型（策略网络）生成一个回答；
- RM 给这个回答打分（奖励信号）；
- PPO 算法用这个奖励去更新模型参数，强化高分路径，压制低分路径；
- 为了防止模型为了骗高分而"走火入魔"，PPO 会引入一个 **KL 散度惩罚项**，保证模型离 SFT 版本不要跑得太远。

**PPO 的 KL 惩罚公式：**

$$
\mathcal{L} = -\mathbb{E}[r_\theta(x, y)] + \beta \cdot \text{KL}[\pi_\theta \| \pi_\text{SFT}]
$$

其中 $r_\theta$ 是奖励分数，$\beta$ 控制对齐约束的松紧程度。

---

## 第二部分：DPO——抛弃奖励模型的对齐捷径

RLHF 虽然强大，但流程复杂：需要同时维护策略模型、参考模型、奖励模型，PPO 本身也是出了名的调参难。

2023 年的论文 **"Direct Preference Optimization: Your Language Model is a Reward Model"** 提出了一个数学上等价但工程上更简洁的方案——**DPO**。

### 2.1 DPO 的核心思路

DPO 的数学推导发现：奖励模型其实可以被**隐式地编码在策略模型自身里**。只需要对比好回答（chosen）和坏回答（rejected）在策略模型 vs 参考模型上的对数概率之差，就能直接优化偏好，**完全不需要显式训练一个奖励模型**。

**DPO 损失函数：**

$$
\mathcal{L}_\text{DPO} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]
$$

其中 $y_w$ 是人类偏好的好答案（chosen），$y_l$ 是被拒绝的差答案（rejected），$\pi_\text{ref}$ 是冻结的参考模型。

### 2.2 MiniMind DPO 源码剖析

打开 `trainer/train_dpo.py`，核心逻辑只有两个函数：

```python
def logits_to_log_probs(logits, labels):
    # 把模型对每个位置的打分表转化成"生成该 token 的对数概率"
    # logits: (batch_size, seq_len, vocab_size)
    # 先 log_softmax 归一化，再用 gather 抠出对应 token 的那一格
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token  # (batch_size, seq_len)


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # 按序列长度归一化，防止长句子自然概率低而造成不公平比较
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    ref_log_probs   = (ref_log_probs   * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # Batch 的前半部分是 chosen，后半部分是 rejected
    batch_size = ref_log_probs.shape[0]
    chosen_ref_lp    = ref_log_probs[:batch_size // 2]
    reject_ref_lp    = ref_log_probs[batch_size // 2:]
    chosen_policy_lp = policy_log_probs[:batch_size // 2]
    reject_policy_lp = policy_log_probs[batch_size // 2:]

    # 核心：计算策略模型和参考模型在 chosen vs rejected 上的对数概率比之差
    pi_logratios  = chosen_policy_lp - reject_policy_lp  # 策略偏好差
    ref_logratios = chosen_ref_lp    - reject_ref_lp     # 参考基线差
    logits = pi_logratios - ref_logratios  # 隐式奖励差值

    # logsigmoid：让 chosen 比 rejected 好的程度越大，损失越小
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

**【面试必考】DPO 训练需要几个模型？**

需要**两个**：

1. **策略模型（policy model）**：正在被训练的，参数会更新；
2. **参考模型（ref model）**：冻结不动的 SFT 模型副本，作为"不能跑偏太远"的基准锚点。

```python
# train_dpo.py 中的关键初始化
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

# 初始化参考模型，完全冻结
ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
ref_model.eval()
ref_model.requires_grad_(False)  # 参考模型永远不接受梯度
```

### 2.3 DPO vs RLHF 对比


| 维度           | RLHF (PPO)                     | DPO                              |
| :------------- | :----------------------------- | :------------------------------- |
| **模型数量**   | 策略模型 + 参考模型 + 奖励模型 | 策略模型 + 参考模型              |
| **训练稳定性** | 较难调参，PPO 是出了名的难伺候 | 稳定，本质是一个监督学习问题     |
| **计算成本**   | 高（三个模型同时在内存里）     | 较低                             |
| **效果**       | 理论上限更高                   | 实践中大多数场景足够用           |
| **数据格式**   | 需要奖励分数                   | 只需要 (chosen, rejected) 偏好对 |

---

## 第三部分：LoRA——用低秩矩阵撬动百亿参数

### 3.1 为什么需要参数高效微调（PEFT）？

全参数 SFT 的成本与预训练接近，对于大多数人和团队来说是不可承受的。但研究者发现一个规律：**大模型在微调时，参数的变化量（ΔW）实际上是低秩的**——也就是说，这些变化可以用两个小矩阵的乘积来近似表示。

这就是 LoRA（Low-Rank Adaptation）的数学直觉。

### 3.2 LoRA 的数学原理

对于一个预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，正常微调是直接更新 $W_0$。

LoRA 的做法是：**冻结 $W_0$ 不动**，旁边并联插入两个小矩阵：

$$
h = W_0 x + \Delta W x = W_0 x + B A x
$$

其中：

- $A \in \mathbb{R}^{r \times k}$（降维矩阵，$r \ll \min(d, k)$）
- $B \in \mathbb{R}^{d \times r}$（升维矩阵）
- $r$ 是 LoRA 的**秩（rank）**，通常取 4、8、16、64

**参数量对比：**

- 原始权重：$d \times k$ 个参数
- LoRA 新增：$r \times k + d \times r = r(d+k)$ 个参数

当 $r=8, d=k=512$ 时，原始参数 $512^2 = 262144$，LoRA 新增仅 $8 \times (512+512) = 8192$，**节省了 97%**！

### 3.3 MiniMind LoRA 源码精讲

打开 `model/model_lora.py`，MiniMind 的 LoRA 实现极其简洁：

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)   # 降维：in → rank
        self.B = nn.Linear(rank, out_features, bias=False)  # 升维：rank → out

        # ⚠️【初始化策略是精髓！】
        # A 用高斯初始化：保证一开始就有随机多样的特征提取能力
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # B 全零初始化：保证训练最开始 LoRA 的输出为零，不破坏预训练权重的效果
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))  # x → 压缩到 rank 维 → 还原到 out 维
```

**【面试高频：为什么 B 要初始化为零？】**

如果 B 也随机初始化，那么训练最开始时 LoRA 分支会输出一个随机噪声，叠加到预训练权重的输出上，会直接破坏模型一开始好不容易学到的知识。B 全零初始化保证 $\Delta W = BA = 0$，**训练起点和原模型完全等价**，然后让梯度从零开始探索最优的微调方向。

**如何把 LoRA 挂载到已有模型上：**

```python
def apply_lora(model, rank=8):
    for name, module in model.named_modules():
        # 只对方形全连接层（通常是 Q、K、V 投影）挂载 LoRA
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 用闭包保存原始 forward，新的 forward = 原始输出 + LoRA 旁路输出
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)  # W₀x + BAx

            module.forward = forward_with_lora
```

### 3.4 LoRA 训练中的参数冻结

`train_lora.py` 中最关键的工程细节——**只让 LoRA 参数接受梯度，原始权重完全冰封**：

```python
apply_lora(model)  # 挂载 LoRA 分支

lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True   # LoRA 参数：开放训练
        lora_params.append(param)
    else:
        param.requires_grad = False  # 原始权重：永远冻结
```

这样 `optimizer = optim.AdamW(lora_params, lr=args.learning_rate)` 只维护 LoRA 参数的动量和方差，**优化器状态占用的显存也同步大幅缩减**，这才是 LoRA 省显存的完整原理（不只是参数少，连优化器状态也少了）。

**LoRA 只保存 LoRA 权重，不保存整个模型：**

```python
def save_lora(model, path):
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            # 只把每个层的 lora 子模块权重打包存储
            lora_state = {f'{clean_name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)  # 保存的文件极小！
```

### 3.5 LoRA 超参数选择指南


| 超参数                | 常见范围             | 选择逻辑                                                          |
| :-------------------- | :------------------- | :---------------------------------------------------------------- |
| **rank（秩）**        | 4 / 8 / 16 / 32 / 64 | 任务越复杂、数据越多，可以用更大的 rank；资源紧张时 4 或 8 已足够 |
| **学习率**            | `1e-4` ~ `1e-3`      | 比全参数 SFT 可以适当大一些，因为只有少量参数在变                 |
| **目标层**            | Q、K、V、O 投影层    | 注意力层通常最重要；也可以加到 FFN 层，但收益递减                 |
| **alpha（缩放系数）** | 通常 = 2 × rank     | 控制 LoRA 输出的缩放幅度，$\Delta W = \frac{\alpha}{r} BA$        |

---

## 第四部分：模型量化——让大模型住进消费级显卡

### 4.1 为什么要量化？

GPT-3 175B 参数，用 FP32（每个参数 4 字节）存储需要 **700 GB** 显存。但一张消费级显卡（RTX 4090）只有 24 GB。量化就是"压缩体积"的核心技术。

### 4.2 量化的本质：用低精度数字近似高精度权重


| 精度格式        | 位数   | 每参数字节 | 动态范围     | 精度损失               |
| :-------------- | :----- | :--------- | :----------- | :--------------------- |
| FP32（全精度）  | 32 bit | 4 字节     | 大           | 无（基准）             |
| BF16（脑浮点）  | 16 bit | 2 字节     | 与 FP32 相当 | 极小                   |
| INT8（8位整型） | 8 bit  | 1 字节     | 小           | 较小，大多数任务可接受 |
| INT4（4位整型） | 4 bit  | 0.5 字节   | 很小         | 明显，需要特殊量化策略 |

**量化的数学操作（以 INT8 为例）：**

$$
x_{\text{int8}} = \text{round}\left(\frac{x_{\text{fp32}}}{\text{scale}}\right), \quad \text{scale} = \frac{\max|x|}{{127}}
$$

反量化时：$x_{\text{dequant}} = x_{\text{int8}} \times \text{scale}$

每次量化+反量化都会引入舍入误差，这就是精度损失的来源。

### 4.3 训练后量化（PTQ）vs 量化感知训练（QAT）


| 方式                                   | 流程                       | 优点                       | 缺点                 |
| :------------------------------------- | :------------------------- | :------------------------- | :------------------- |
| **PTQ（Post-Training Quantization）**  | 训练完成后直接对权重做量化 | 无需重新训练，简单快速     | 精度损失相对大       |
| **QAT（Quantization-Aware Training）** | 训练过程中模拟量化误差     | 精度损失小，效果接近全精度 | 需要重新训练，成本高 |

**主流工具：**

- **GGUF/GGML（llama.cpp）**：最流行的本地推理量化格式，支持 Q4_K_M、Q8_0 等多种精度，让 7B 模型能在 4GB 显存甚至纯 CPU 上流畅运行；
- **bitsandbytes**：PyTorch 训练时的 INT8/NF4 量化库，支持 QLoRA（量化 + LoRA 结合的极致省显存方案）；
- **AWQ / GPTQ**：面向 GPU 推理的高精度量化方案，在 INT4 精度下仍能保持接近 FP16 的效果。

### 4.4 QLoRA：量化 + LoRA 的终极省显存组合

QLoRA = 将基础模型量化为 **NF4（4-bit 正态浮点）** 精度 + 在量化后的模型上挂载 LoRA 适配器。

效果：在单张 24GB 显卡上微调 65B 参数的模型——这在之前是不可想象的。代价是训练速度比全精度慢约 30%，但效果损失极小。

---

## 第五部分：今日知识图谱与面试拷问

### 5.1 本日核心概念连接图



```
预训练（大量无标注数据）
    ↓
SFT（少量指令数据，全参数或 LoRA）
    ↓
对齐优化（选其一）
    ├── RLHF：SFT → RM 训练 → PPO 强化学习
    └── DPO：SFT → 直接偏好优化（更简单）
    ↓
推理部署（量化压缩：FP16 → INT8/INT4）
```

### 5.2 面试核心拷问

**Q：LoRA 的 rank 越大越好吗？**
不一定。rank 越大，可训练参数越多，拟合能力越强，但也越容易过拟合（尤其是数据量不足时），且省显存的优势减弱。实践中 rank=8 或 16 对大多数任务已经足够，需要根据数据量和任务复杂度权衡。

**Q：DPO 中的 beta 参数是做什么的？**
$\beta$ 控制 KL 约束的强度，决定策略模型允许偏离参考模型多远。$\beta$ 越大，模型被约束得越紧（保守），不容易过拟合偏好数据；$\beta$ 越小，模型对偏好数据的拟合越积极，但风险是"走火入魔"、输出奇怪。MiniMind 默认 `beta=0.1`，这是业界常用值。

**Q：量化为什么会损失精度？如何降低损失？**
量化把连续的浮点数映射到有限的离散整数格，必然存在舍入误差。降低损失的方法：

1. **用更好的量化粒度**：按行/按通道而不是按整层量化（如 AWQ 的逐组量化）；
2. **校准数据集**：用少量真实数据做 PTQ 校准，让 scale 因子更准确；
3. **QAT**：训练时模拟量化误差，让模型学会在量化精度下工作；
4. **混合精度量化**：对精度敏感的层保持 FP16/INT8，其余层用 INT4。

**Q：LoRA 和全参数 SFT 相比，最终效果一定差吗？**
不一定。对于数据量有限的下游任务，LoRA 适度的正则化反而可以防止过拟合，效果可能持平甚至超过全参数 SFT。但如果数据量极大、任务需要大幅改变模型行为，全参数微调的上限更高。

---

## 🚀 上机实战任务

> 今天的任务是把 LoRA 微调跑起来，感受它和 Full SFT 的差异：
>
> ```bash
> # 先确认有 SFT 权重（LoRA 是在 SFT 之上微调的）
> python trainer/train_lora.py \
>   --lora_name lora_identity \
>   --epochs 5 \
>   --batch_size 8 \
>   --learning_rate 1e-4 \
>   --from_weight full_sft
> ```
>
> 观察：
>
> 1. 打印出的 `LoRA 参数量` 和 `LLM 总参数量` 的比值（通常在 1%~5%）；
> 2. 对比 LoRA 微调前后，模型对同一个问题的回答是否发生了变化；
> 3. `out/lora/` 目录下保存的 `.pth` 文件大小，对比 `out/` 下全参数权重的大小——你会发现差距可能在 100 倍以上。

---

> 准备好了吗？第四天，我们将进入大模型的**推理加速战场**：KV Cache 的显存-速度权衡，Flash Attention 如何用 IO 优化把注意力计算从 $O(N^2)$ 的显存爆炸中拯救出来，以及多卡并行的分布式推理策略——速度与规模的最终博弈！
