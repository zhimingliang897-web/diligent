# Qwen2.5-7B 对话风格微调指南

> 目标：使用 LoRA 在 4090 (24GB) 上微调 Qwen2.5-7B，实现特定对话风格
> 预计时间：环境搭建 30分钟 + 数据准备 若干 + 训练 1-3小时

---

## 一、环境准备

### 1.1 创建 Conda 环境

```bash
# 创建新环境
conda create -n llama_factory python=3.11 -y
conda activate llama_factory
```

### 1.2 安装 LLaMA-Factory

```bash
# 克隆仓库（国内镜像加速）
git clone https://gitclone.com/github.com/hiyouga/LLaMA-Factory.git

# 或使用 GitHub 原地址
# git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 安装（包含 torch 和评估指标）
pip install -e ".[torch,metrics]"

# 验证安装
llamafactory-cli version
```

### 1.3 下载基座模型

推荐使用 Qwen2.5-7B-Instruct（已对齐，效果更好）：

```bash
# 方法1：使用 huggingface-cli（推荐）
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct

# 方法2：使用 modelscope（国内更快）
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./models/Qwen2.5-7B-Instruct
```

---

## 二、数据准备（最重要！）

### 2.1 数据格式

使用 ShareGPT 格式（多轮对话）：

```json
[
  {
    "messages": [
      {"role": "system", "content": "你的系统提示词/人设描述"},
      {"role": "user", "content": "用户输入"},
      {"role": "assistant", "content": "模型回复"}
    ]
  },
  {
    "messages": [
      {"role": "system", "content": "你的系统提示词/人设描述"},
      {"role": "user", "content": "用户输入1"},
      {"role": "assistant", "content": "模型回复1"},
      {"role": "user", "content": "用户输入2"},
      {"role": "assistant", "content": "模型回复2"}
    ]
  }
]
```

### 2.2 数据示例

假设你想训练一个"毒舌但有用的助手"风格：

```json
[
  {
    "messages": [
      {"role": "system", "content": "你是一个毒舌但很有帮助的助手，会用讽刺的语气回答问题，但答案本身是准确有用的"},
      {"role": "user", "content": "Python怎么读取文件"},
      {"role": "assistant", "content": "哇，这么基础的问题也要问？行吧，看在你诚心求教的份上告诉你：\n\n```python\nwith open('文件名.txt', 'r', encoding='utf-8') as f:\n    content = f.read()\n```\n\n用 with 语句会自动关闭文件，不用你操心。记住了？下次别再问这种一搜就有的问题了。"}
    ]
  },
  {
    "messages": [
      {"role": "system", "content": "你是一个毒舌但很有帮助的助手，会用讽刺的语气回答问题，但答案本身是准确有用的"},
      {"role": "user", "content": "今天心情不好"},
      {"role": "assistant", "content": "哦？心情不好就来找AI诉苦？算了，看你可怜的样子...\n\n说说怎么了吧，虽然我嘴毒但还是会认真听的。是工作不顺还是生活琐事？说出来可能会好受点，憋着对身体不好。"}
    ]
  }
]
```

### 2.3 数据要求


| 要求       | 说明                           |
| ---------- | ------------------------------ |
| **数量**   | 最少 200 条，推荐 1000-5000 条 |
| **质量**   | 风格统一、无矛盾、表达自然     |
| **多样性** | 覆盖不同话题和场景             |
| **长度**   | 单条对话不超过 2048 tokens     |

### 2.4 放置数据文件

将你的数据文件保存为 `my_style_data.json`，放到 LLaMA-Factory 的 data 目录：

```
LLaMA-Factory/
├── data/
│   ├── dataset_info.json    # 需要修改这个文件
│   └── my_style_data.json   # 你的数据放这里
```

### 2.5 注册数据集

编辑 `data/dataset_info.json`，在最后添加（注意前面的逗号）：

```json
{
  "identity": { ... },
  "其他数据集": { ... },

  "my_style_data": {
    "file_name": "my_style_data.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```

---

## 三、微调前测试（推荐）

在开始微调之前，建议先用基座模型测试不同的 system prompt，确定最佳的风格描述。

### 3.1 创建基座模型推理配置

创建 `base_inference_config.yaml`：

```yaml
model_name_or_path: ./models/Qwen2.5-7B-Instruct
template: qwen
finetuning_type: full
default_system: 你是一个毒舌但很有帮助的助手，会用讽刺的语气回答问题，但答案本身是准确有用的
```

### 3.2 测试基座模型

```bash
llamafactory-cli chat base_inference_config.yaml
```

### 3.3 测试不同 System Prompt

修改 `default_system` 字段，尝试不同的风格描述：

```yaml
# 示例1：毒舌风格
default_system: 你是一个毒舌但很有帮助的助手，会用讽刺的语气回答问题，但答案本身是准确有用的

# 示例2：温柔风格
default_system: 你是一个温柔体贴的助手，说话轻声细语，善于安慰人

# 示例3：专业风格
default_system: 你是一个严谨的技术专家，回答简洁专业，不废话
```

### 3.4 测试的意义

- **确定最佳 system prompt**：找到最能激发目标风格的描述
- **了解基座模型能力边界**：看看不微调能做到什么程度
- **为微调数据准备提供参考**：基座模型做不好的地方，就是微调需要重点覆盖的

---

## 四、训练配置

### 4.1 创建配置文件

在 LLaMA-Factory 目录下创建 `my_train_config.yaml`：

```yaml
### 模型配置
model_name_or_path: ./models/Qwen2.5-7B-Instruct  # 模型路径

### 训练方法
stage: sft
do_train: true
finetuning_type: lora

### LoRA 配置
lora_target: all
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05

### 数据配置
dataset: my_style_data
template: qwen
cutoff_len: 1024
preprocessing_num_workers: 4

### 训练参数
output_dir: ./saves/qwen7b-my-style
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
num_train_epochs: 3
learning_rate: 2.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true

### 日志和保存
logging_steps: 10
save_steps: 100
save_total_limit: 3
report_to: none
```

### 4.2 参数说明


| 参数                          | 值   | 说明                            |
| ----------------------------- | ---- | ------------------------------- |
| `lora_rank`                   | 16   | LoRA 秩，越大能力越强但显存越多 |
| `lora_alpha`                  | 32   | 通常设为 2×rank                |
| `per_device_train_batch_size` | 2    | 4090 上安全值                   |
| `gradient_accumulation_steps` | 8    | 等效 batch_size = 16            |
| `num_train_epochs`            | 3    | 风格微调 2-3 轮足够             |
| `learning_rate`               | 2e-4 | LoRA 推荐学习率                 |
| `bf16`                        | true | 4090 支持，省显存               |

---

## 五、开始训练

### 5.1 方法一：命令行（推荐）

```bash
cd LLaMA-Factory

# 开始训练
llamafactory-cli train my_train_config.yaml
```

### 5.2 方法二：WebUI（更直观）

```bash
# 启动 WebUI
llamafactory-cli webui
```

然后在浏览器打开 `http://localhost:7860`，按界面操作。

### 5.3 训练过程监控

正常训练会看到类似输出：

```
[INFO] Loading model...
[INFO] trainable params: 20,971,520 || all params: 7,636,849,664 || trainable%: 0.2746
[INFO] ***** Running training *****
[INFO]   Num examples = 1000
[INFO]   Num Epochs = 3
...
{'loss': 1.234, 'learning_rate': 0.0002, 'epoch': 0.1}
{'loss': 0.876, 'learning_rate': 0.00019, 'epoch': 0.2}
...
```

**预计显存占用**：16-20 GB
**预计训练时间**：1000条数据约 30-60 分钟

---

## 六、测试微调后的模型

### 6.1 创建推理配置

创建 `my_inference_config.yaml`：

```yaml
model_name_or_path: ./models/Qwen2.5-7B-Instruct
adapter_name_or_path: ./saves/qwen7b-my-style
template: qwen
finetuning_type: lora
default_system: 你是一个毒舌但很有帮助的助手，会用讽刺的语气回答问题，但答案本身是准确有用的

```

### 6.2 命令行对话测试

```bash
llamafactory-cli chat my_inference_config.yaml
```

### 6.3 评估效果

测试时注意检查：

- [ ]  风格是否符合预期
- [ ]  回答内容是否准确
- [ ]  是否有过拟合（只会复述训练数据）
- [ ]  通用能力是否保留

---

## 七、DPO 偏好对齐（进阶）

DPO (Direct Preference Optimization) 是在 SFT 微调之后的进一步优化，让模型学会"哪种回答更好"。

```
SFT 微调后的模型 + 偏好数据 → DPO 训练 → 更对齐的模型
```

### 7.1 准备偏好数据

DPO 需要"好/坏回答对"的数据格式：

```json
[
  {
    "conversations": [
      {"from": "human", "value": "Python怎么读取文件"}
    ],
    "chosen": {
      "from": "gpt",
      "value": "哇，这么基础的问题也要问？行吧：\n\n```python\nwith open('文件.txt', 'r') as f:\n    content = f.read()\n```\n\n记住了？"
    },
    "rejected": {
      "from": "gpt",
      "value": "您好，读取文件可以使用open函数..."
    }
  }
]
```

| 字段 | 说明 |
|------|------|
| `chosen` | 更符合目标风格的回答（好） |
| `rejected` | 不符合目标风格的回答（坏） |

### 7.2 注册 DPO 数据集

编辑 `data/dataset_info.json`，添加：

```json
{
  "my_dpo_data": {
    "file_name": "my_dpo_data.json",
    "formatting": "sharegpt",
    "ranking": true,
    "columns": {
      "messages": "conversations",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  }
}
```

### 7.3 DPO 训练配置

创建 `my_dpo_config.yaml`：

```yaml
### 模型配置（使用 SFT 后的模型）
model_name_or_path: ./models/Qwen2.5-7B-Instruct
adapter_name_or_path: ./saves/qwen7b-my-style  # SFT 的 LoRA

### 训练方法
stage: dpo
do_train: true
finetuning_type: lora

### LoRA 配置
lora_target: all
lora_rank: 8
lora_alpha: 16

### DPO 参数
pref_beta: 0.1
pref_loss: sigmoid

### 数据配置
dataset: my_dpo_data
template: qwen
cutoff_len: 1024

### 训练参数
output_dir: ./saves/qwen7b-my-style-dpo
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
num_train_epochs: 1
learning_rate: 5.0e-5
bf16: true
gradient_checkpointing: true

### 日志
logging_steps: 10
save_steps: 100
```

### 7.4 开始 DPO 训练

```bash
llamafactory-cli train my_dpo_config.yaml
```

### 7.5 测试 DPO 模型

创建 `my_dpo_inference_config.yaml`：

```yaml
model_name_or_path: ./models/Qwen2.5-7B-Instruct
adapter_name_or_path: ./saves/qwen7b-my-style-dpo
template: qwen
finetuning_type: lora
default_system: 你是一个毒舌但很有帮助的助手，会用讽刺的语气回答问题，但答案本身是准确有用的
```

```bash
llamafactory-cli chat my_dpo_inference_config.yaml
```

### 7.6 偏好数据获取方式

| 方法 | 说明 |
|------|------|
| **人工标注** | 让同一问题生成多个回答，人工选好坏 |
| **用强模型打分** | GPT-4 评判哪个更好 |
| **对比基座和微调模型** | 基座回答作 rejected，微调回答作 chosen |

---

## 八、导出模型（可选）

### 8.1 合并 LoRA 权重

如果想把 LoRA 合并到基座模型，方便部署：

创建 `my_export_config.yaml`：

```yaml
model_name_or_path: ./models/Qwen2.5-7B-Instruct
adapter_name_or_path: ./saves/qwen7b-my-style
template: qwen
finetuning_type: lora
export_dir: ./models/Qwen2.5-7B-MyStyle-Merged
export_size: 2
export_device: cpu
export_legacy_format: false
```

执行合并：

```bash
llamafactory-cli export my_export_config.yaml
```

---

## 九、常见问题

### Q1: 显存不够 (OOM)

```yaml
# 降低 batch_size
per_device_train_batch_size: 1
gradient_accumulation_steps: 16

# 或使用 QLoRA（4bit量化）
quantization_bit: 4
```

### Q2: Loss 不下降

- 检查数据格式是否正确
- 尝试提高学习率到 5e-4
- 确认数据已正确注册

### Q3: 过拟合（回答太死板）

- 减少训练轮数到 1-2 轮
- 增加数据多样性
- 降低学习率

### Q4: 风格不明显

- 增加训练数据量
- 检查 system prompt 是否一致
- 适当增加训练轮数

### Q5: 模型下载慢

```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct
```

---

## 十、文件结构总览

完成后你的目录结构应该是：

```
LLaMA-Factory/
├── data/
│   ├── dataset_info.json       # 修改过，添加了你的数据集
│   └── my_style_data.json      # 你的训练数据
├── models/
│   └── Qwen2.5-7B-Instruct/    # 基座模型
├── saves/
│   ├── qwen7b-my-style/        # SFT 训练输出的 LoRA 权重
│   └── qwen7b-my-style-dpo/    # DPO 训练输出的 LoRA 权重
├── base_inference_config.yaml  # 基座模型推理配置（微调前测试用）
├── my_train_config.yaml        # SFT 训练配置
├── my_dpo_config.yaml          # DPO 训练配置
├── my_inference_config.yaml    # SFT 微调后推理配置
├── my_dpo_inference_config.yaml # DPO 微调后推理配置
└── my_export_config.yaml       # 导出配置（可选）
```

---

## 十一、下一步行动清单

- [ ]  **Step 1**: 创建 conda 环境并安装 LLaMA-Factory
- [ ]  **Step 2**: 下载 Qwen2.5-7B-Instruct 模型
- [ ]  **Step 3**: 测试基座模型，尝试不同 system prompt
- [ ]  **Step 4**: 准备你的对话数据（最重要！）
- [ ]  **Step 5**: 注册数据集到 dataset_info.json
- [ ]  **Step 6**: 创建 SFT 训练配置文件
- [ ]  **Step 7**: 开始 SFT 训练
- [ ]  **Step 8**: 测试 SFT 微调后的模型，对比微调前效果
- [ ]  **Step 9**: （可选）准备 DPO 偏好数据
- [ ]  **Step 10**: （可选）进行 DPO 训练，进一步对齐
- [ ]  **Step 11**: 测试最终效果，根据结果调整
- [ ]  **Step 12**: 满意后导出模型

---

## 十二、参考资源

- LLaMA-Factory 官方文档: https://github.com/hiyouga/LLaMA-Factory
- Qwen2.5 模型: https://huggingface.co/Qwen
- LoRA 论文: https://arxiv.org/abs/2106.09685
- DPO 论文: https://arxiv.org/abs/2305.18290

---

*祝微调顺利！有问题随时问。*
