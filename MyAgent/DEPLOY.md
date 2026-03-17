# MyAgent 云服务器部署指南

## 服务器信息

| 项目 | 值 |
|------|-----|
| IP | 8.138.164.133 |
| 域名 | api.liangyiren.top |
| 系统 | Alibaba Cloud Linux 8 |
| Python | 3.11 (miniconda3) |
| 内存 | 1.8GB |

## 访问地址

| 服务 | 地址 |
|------|------|
| **主页 (Integrity Tools)** | `http://8.138.164.133/` |
| **MyAgent 智能体** | `http://8.138.164.133/agent/` |
| **AI 辩论赛** | `http://8.138.164.133/debate/` |
| **PDF 工具** | `http://8.138.164.133/pdf/` |
| **File Agent** | `http://8.138.164.133/files/` |
| **Video 下载** | `http://8.138.164.133/video/` |

> **注意**：服务器已移除 HTTPS 强制跳转，统一使用 HTTP 访问。如果浏览器仍跳转 HTTPS，请在 Chrome 地址栏输入 `chrome://net-internals/#hsts`，删除 `api.liangyiren.top` 的安全策略后重试。

## 目录结构

```
/root/myagent/
├── webui.py              # Web 入口
├── main.py               # CLI 入口
├── requirements.txt      # 依赖清单
├── .env                  # API Key 配置
├── agent/                # 核心模块
│   ├── config.py
│   ├── graph.py
│   ├── llm.py
│   ├── tools.py
│   ├── callbacks.py
│   ├── rag/
│   ├── memory/
│   └── multi/
└── scripts/
```

## 部署步骤

### 1. 创建 Conda 环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n myagent python=3.11 -y
conda activate myagent
```

### 2. 上传代码

从本地上传到服务器：
```bash
scp -r /e/diligent/MyAgent/* root@8.138.164.133:/root/myagent/
```

### 3. 安装依赖

```bash
conda activate myagent
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  langchain langchain-community langchain-core langchain-text-splitters \
  langgraph langgraph-checkpoint-sqlite dashscope faiss-cpu \
  python-dotenv httpx python-docx openpyxl python-pptx rank_bm25 gradio ddgs
```

### 4. 配置环境变量

```bash
cat > /root/myagent/.env << EOF
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
EOF
```

### 5. 创建 systemd 服务

```bash
cat > /etc/systemd/system/myagent.service << EOF
[Unit]
Description=MyAgent AI Assistant
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/myagent
Environment="PATH=/root/miniconda3/envs/myagent/bin"
ExecStart=/root/miniconda3/envs/myagent/bin/python webui.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable myagent
systemctl start myagent
```

### 6. 配置 Nginx

在 `/etc/nginx/conf.d/api.liangyiren.top.conf` 中添加：

```nginx
upstream myagent {
    server 127.0.0.1:7860;
}

# 在 server 块内添加
location /agent/ {
    rewrite ^/agent/(.*) /$1 break;
    proxy_pass http://myagent;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 86400s;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

重载 Nginx：
```bash
nginx -t && systemctl reload nginx
```

### 7. 修改 webui.py

确保 `webui.py` 中的启动配置为：
```python
demo.launch(
    root_path="/agent",
    server_name="0.0.0.0",
    server_port=7860,
    ...
)
```

## 管理命令

```bash
# 查看状态
systemctl status myagent

# 查看日志
journalctl -u myagent -f

# 重启服务
systemctl restart myagent

# 停止服务
systemctl stop myagent

# 启动服务
systemctl start myagent
```

## 端口分配

| 服务 | 端口 | systemd 服务名 |
|------|------|----------------|
| AI辩论赛 | 5001 | ai-debate |
| PDF工具 | 5002 | pdf-tools |
| 2台词 | 5003 | lines-service |
| File Agent | 5004 | file-agent |
| Video下载 | 5005 | video-downloader |
| Integrity Tools | 5006 | integrity-tools |
| **MyAgent** | **7860** | **myagent** |

## 知识库配置（可选）

如需启用 RAG 知识库：

```bash
# 1. 上传知识库文档
scp -r /e/diligent/MyAgent/data/ root@8.138.164.133:/root/myagent/
scp -r /e/diligent/MyAgent/vectorstore/ root@8.138.164.133:/root/myagent/

# 2. 重启服务
systemctl restart myagent
```

## 故障排查

### 服务无法启动

```bash
# 检查日志
journalctl -u myagent -n 50

# 手动运行测试
cd /root/myagent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate myagent
python webui.py
```

### 页面无法访问

```bash
# 检查端口
netstat -tlnp | grep 7860

# 检查 Nginx
nginx -t
systemctl status nginx

# 测试本地访问
curl http://127.0.0.1:7860/
```

### API Key 问题

```bash
# 检查配置
cat /root/myagent/.env

# 测试 API
curl -X POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-plus","messages":[{"role":"user","content":"hi"}]}'
```

## 更新部署

```bash
# 1. 上传新代码
scp /e/diligent/MyAgent/webui.py root@8.138.164.133:/root/myagent/
scp -r /e/diligent/MyAgent/agent/ root@8.138.164.133:/root/myagent/

# 2. 重启服务
systemctl restart myagent
```

---

*最后更新: 2026-03-17*