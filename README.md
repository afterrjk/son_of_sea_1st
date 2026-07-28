# 筑安

“筑安”是面向建筑工地一线产业工人的 AI 安全生产助手。本阶段完成 FastAPI、
DeepSeek Anthropic 兼容接口、SSE 流式输出和单文件演示前端的基础工程初始化。

## 目录结构

```text
son_of_sea/
├── app/
│   ├── __init__.py
│   └── deepseek_client.py
├── data/
│   └── knowledge.json
├── static/
│   └── index.html
├── .env
├── .env.example
├── .gitignore
├── main.py
├── requirements.txt
└── README.md
```

## 本地运行

在 PowerShell 中执行：

```powershell
cd "E:\AI agent\son_of_sea"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload
```

浏览器打开 <http://127.0.0.1:8000>。接口文档位于
<http://127.0.0.1:8000/docs>，健康检查位于 <http://127.0.0.1:8000/health>。

## 对话接口

`POST /chat` 请求示例：

```json
{
  "message": "高处作业前要检查什么？",
  "history": [
    {"role": "user", "content": "我在做外墙施工。"},
    {"role": "assistant", "content": "请问作业高度和脚手架情况？"}
  ]
}
```

响应类型为 `text/event-stream`，主要事件如下：

```text
data: {"type":"delta","content":"首先"}

data: {"type":"done"}
```

## 安全说明

- `.env` 已被 Git 忽略，不要把真实 API Key 写进代码或提交到仓库。
- `data/knowledge.json` 当前只有演示占位内容，不应直接作为生产安全依据。
- ChromaDB 和句向量依赖已加入环境，知识切分、入库与检索将在 RAG 阶段实现。

