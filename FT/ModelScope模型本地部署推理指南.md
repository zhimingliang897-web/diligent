# ModelScope 模型本地部署推理指南

本指南介绍如何从 ModelScope（魔搭社区）下载模型并在本地进行推理。以 LLaMA-2-7B 为例。

> **优势**：ModelScope 是国内平台，下载速度快，无需申请权限，无需科学上网。

## 1. 环境准备

```bash
# 安装必要的库
pip install modelscope
pip install transformers torch accelerate
```

## 2. 下载模型

### 方法一：使用 modelscope CLI（推荐）

```bash
# 下载模型到默认缓存目录
modelscope download --model modelscope/Llama-2-7b-chat-ms

# 下载到指定目录
modelscope download --model modelscope/Llama-2-7b-chat-ms --local_dir ./models/Llama-2-7b-chat-ms
```

### 方法二：使用 Python 下载

```python
from modelscope import snapshot_download

# 下载模型到指定目录
model_dir = snapshot_download(
    'modelscope/Llama-2-7b-chat-ms',
    cache_dir='./models'
)
print(f"模型下载到: {model_dir}")
```

### 方法三：使用 Git LFS

```bash
# 安装 git-lfs
apt install git-lfs
git lfs install

# 克隆模型仓库
git clone https://www.modelscope.cn/modelscope/Llama-2-7b-chat-ms.git ./models/Llama-2-7b-chat-ms
```

## 3. 本地推理

### 方法一：使用 ModelScope 原生推理

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载模型和分词器
model_path = "./models/Llama-2-7b-chat-ms"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 构建对话
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

### 方法二：使用 Transformers 推理

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./models/Llama-2-7b-chat-ms"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 简单推理
prompt = "What is machine learning?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 方法三：使用 LLaMA-Factory 推理

1. **创建推理配置文件**

```bash
cat > my_inference_config.yaml << 'EOF'
model_name_or_path: ./models/Llama-2-7b-chat-ms
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
model_name_or_path: ./models/Llama-2-7b-chat-ms
adapter_name_or_path: ./saves/llama2-7b/lora/train1
template: llama2
finetuning_type: lora
EOF

llamafactory-cli chat my_lora_inference_config.yaml
```

### 方法四：使用 vLLM 高性能推理

```bash
# 安装 vLLM
pip install vllm

# 启动 API 服务
python -m vllm.entrypoints.openai.api_server \
    --model ./models/Llama-2-7b-chat-ms \
    --trust-remote-code \
    --port 8000
```

使用 API：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="./models/Llama-2-7b-chat-ms",
    messages=[
        {"role": "user", "content": "Hello"}
    ]
)
print(response.choices[0].message.content)
```

## 4. ModelScope 常用模型

| 模型 | ModelScope 路径 | 大小 | 说明 |
|------|-----------------|------|------|
| Llama-2-7B-Chat | modelscope/Llama-2-7b-chat-ms | ~14GB | LLaMA 2 对话版 |
| Llama-2-13B-Chat | modelscope/Llama-2-13b-chat-ms | ~26GB | 更大版本 |
| Llama-3-8B-Instruct | LLM-Research/Meta-Llama-3-8B-Instruct | ~16GB | LLaMA 3 |
| Qwen2.5-7B-Instruct | Qwen/Qwen2.5-7B-Instruct | ~14GB | 通义千问 |
| Qwen2.5-14B-Instruct | Qwen/Qwen2.5-14B-Instruct | ~28GB | 更大版本 |
| Yi-1.5-9B-Chat | 01ai/Yi-1.5-9B-Chat | ~18GB | 零一万物 |
| DeepSeek-V2-Lite-Chat | deepseek-ai/DeepSeek-V2-Lite-Chat | ~32GB | DeepSeek |
| GLM-4-9B-Chat | ZhipuAI/glm-4-9b-chat | ~18GB | 清华 GLM |
| Mistral-7B-Instruct | AI-ModelScope/Mistral-7B-Instruct-v0.3 | ~14GB | Mistral |

## 5. HuggingFace vs ModelScope 对比

| 对比项 | HuggingFace | ModelScope |
|--------|-------------|------------|
| 下载速度（国内） | 较慢，需镜像 | 快 |
| 是否需要申请 | 部分模型需要 | 大部分不需要 |
| 模型数量 | 更多 | 较少但主流都有 |
| 科学上网 | 可能需要 | 不需要 |

## 6. 常见问题

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

### Q: 如何查看模型的 template？

在 LLaMA-Factory 中，常用 template：
- LLaMA-2：`llama2`
- LLaMA-3：`llama3`
- Qwen 系列：`qwen`
- Mistral：`mistral`
- Yi：`yi`
- GLM：`glm4`
- DeepSeek：`deepseek`

### Q: 如何设置 ModelScope 缓存目录？

```bash
# 设置环境变量
export MODELSCOPE_CACHE=/your/cache/path

# 或在 Python 中设置
import os
os.environ['MODELSCOPE_CACHE'] = '/your/cache/path'
```

### Q: 如何搜索 ModelScope 上的模型？

访问 https://www.modelscope.cn/models 搜索你需要的模型。

## 7. 快速开始示例

下载并运行 LLaMA-2-7B 的完整流程：

```bash
# 1. 安装依赖
pip install modelscope transformers torch accelerate

# 2. 下载模型
modelscope download --model modelscope/Llama-2-7b-chat-ms --local_dir ./models/Llama-2-7b-chat-ms

# 3. 创建配置文件
cat > inference_config.yaml << 'EOF'
model_name_or_path: ./models/Llama-2-7b-chat-ms
template: llama2
EOF

# 4. 启动对话
llamafactory-cli chat inference_config.yaml
```

## 8. 使用 ModelScope 下载 HuggingFace 模型

ModelScope 还可以作为 HuggingFace 的镜像使用：

```python
# 设置环境变量使用 ModelScope 镜像
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 然后正常使用 HuggingFace 的方式下载
from huggingface_hub import snapshot_download
snapshot_download("meta-llama/Llama-2-7b-chat-hf", local_dir="./models/Llama-2-7b-chat-hf")
```

或者直接使用 ModelScope SDK 下载 HuggingFace 模型：

```python
from modelscope import snapshot_download

# 部分 HuggingFace 模型在 ModelScope 上有镜像
model_dir = snapshot_download('modelscope/Llama-2-7b-chat-ms', cache_dir='./models')
```
