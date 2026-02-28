# HuggingFace 模型本地部署推理指南

本指南介绍如何从 HuggingFace 下载模型并在本地进行推理。以 LLaMA-2-7B 为例。

## 1. 环境准备

```bash
# 安装必要的库
pip install transformers torch accelerate
pip install huggingface_hub
```

## 2. 下载模型

> **注意**：LLaMA 系列模型需要先在 HuggingFace 上申请访问权限，并登录账号。

### 方法一：使用 huggingface-cli（推荐）

```bash
# 登录 HuggingFace（LLaMA 模型必须登录）
huggingface-cli login

# 下载模型到指定目录
huggingface-cli download <模型名称> --local-dir ./models/<模型名称>

# 示例：下载 LLaMA-2-7B-Chat
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir ./models/Llama-2-7b-chat-hf
```

### 方法二：使用 Python 下载

```python
from huggingface_hub import snapshot_download

# 下载模型
snapshot_download(
    repo_id="meta-llama/Llama-2-7b-chat-hf",
    local_dir="./models/Llama-2-7b-chat-hf"
)
```

### 方法三：使用 Git LFS

```bash
# 安装 git-lfs
apt install git-lfs
git lfs install

# 克隆模型仓库
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf ./models/Llama-2-7b-chat-hf
```

## 3. 本地推理

### 方法一：使用 Transformers 原生推理

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载模型和分词器
model_path = "./models/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 或 torch.bfloat16
    device_map="auto",          # 自动分配到 GPU
    trust_remote_code=True
)

# 构建对话（LLaMA-2 格式）
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, please introduce yourself."}
]

# 生成回复
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

### 方法二：使用 LLaMA-Factory 推理

1. **创建推理配置文件**

```bash
cat > my_inference_config.yaml << 'EOF'
model_name_or_path: ./models/Llama-2-7b-chat-hf
template: llama2
EOF
```

2. **启动交互式对话**

```bash
llamafactory-cli chat my_inference_config.yaml
```

3. **如果需要加载 LoRA 适配器**

```bash
cat > my_lora_inference_config.yaml << 'EOF'
model_name_or_path: ./models/Llama-2-7b-chat-hf
adapter_name_or_path: ./saves/llama2-7b/lora/train1
template: llama2
finetuning_type: lora
EOF

llamafactory-cli chat my_lora_inference_config.yaml
```

### 方法三：使用 vLLM 高性能推理

```bash
# 安装 vLLM
pip install vllm

# 启动 API 服务
python -m vllm.entrypoints.openai.api_server \
    --model ./models/Llama-2-7b-chat-hf \
    --trust-remote-code \
    --port 8000
```

使用 API：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="./models/Llama-2-7b-chat-hf",
    messages=[
        {"role": "user", "content": "Hello"}
    ]
)
print(response.choices[0].message.content)
```

## 4. 常用模型推荐

| 模型 | HuggingFace 路径 | 大小 | 说明 |
|------|------------------|------|------|
| Llama-2-7B-Chat | meta-llama/Llama-2-7b-chat-hf | ~14GB | Meta LLaMA 2，需申请 |
| Llama-2-13B-Chat | meta-llama/Llama-2-13b-chat-hf | ~26GB | 更大版本 |
| Llama-3-8B-Instruct | meta-llama/Meta-Llama-3-8B-Instruct | ~16GB | LLaMA 3，需申请 |
| Qwen2.5-7B-Instruct | Qwen/Qwen2.5-7B-Instruct | ~14GB | 通义千问，中文优秀 |
| Mistral-7B-Instruct | mistralai/Mistral-7B-Instruct-v0.3 | ~14GB | 高效模型 |
| Yi-1.5-9B-Chat | 01-ai/Yi-1.5-9B-Chat | ~18GB | 零一万物 |
| DeepSeek-V2-Lite-Chat | deepseek-ai/DeepSeek-V2-Lite-Chat | ~32GB | DeepSeek |
| GLM-4-9B-Chat | THUDM/glm-4-9b-chat | ~18GB | 清华 GLM |

## 5. 常见问题

### Q: 显存不足怎么办？

```python
# 使用 4-bit 量化
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto"
)
```

### Q: 下载速度慢？

```bash
# 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 然后正常下载
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir ./models/Llama-2-7b-chat-hf
```

### Q: LLaMA 模型需要申请权限？

1. 访问 https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
2. 点击 "Access repository" 申请访问
3. 等待 Meta 批准（通常几小时内）
4. 使用 `huggingface-cli login` 登录后即可下载

### Q: 如何查看模型的 template？

在 LLaMA-Factory 中，常用 template：
- LLaMA-2：`llama2`
- LLaMA-3：`llama3`
- Qwen 系列：`qwen`
- Mistral：`mistral`
- Yi：`yi`
- GLM：`glm4`
- DeepSeek：`deepseek`

## 6. 快速开始示例

下载并运行 LLaMA-2-7B 的完整流程：

```bash
# 1. 登录 HuggingFace（必须，LLaMA 需要权限）
huggingface-cli login

# 2. 设置镜像（可选，国内加速）
export HF_ENDPOINT=https://hf-mirror.com

# 3. 下载模型
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir ./models/Llama-2-7b-chat-hf

# 4. 创建配置文件
cat > inference_config.yaml << 'EOF'
model_name_or_path: ./models/Llama-2-7b-chat-hf
template: llama2
EOF

# 5. 启动对话
llamafactory-cli chat inference_config.yaml
```
