# 筑安 · 建筑工地安全生产 AI 智能助手

面向建筑工地一线产业工人的 AI 安全生产助手，提供安全问答、隐患识别、技能微课和应急指南四大功能模块。

- 技术栈：FastAPI + DeepSeek（Anthropic 兼容 API）+ ChromaDB RAG + sentence-transformers
- 前端：单文件 HTML（原生 JS），SSE 流式对话，结构化结果卡片展示
- 兼容 Python 3.10+，已在 Python 3.12 验证

## 功能模块

| 模块 | 接口 | 说明 |
|---|---|---|
| 安全问答 | `POST /chat` | RAG 检索增强 + SSE 流式回答 |
| 隐患识别 | `POST /analyze_hazard` | 描述现场情况，输出风险等级与处置建议（JSON） |
| 技能微课 | `POST /generate_lesson` | 按工种生成结构化微课（JSON） |
| 应急指南 | `POST /emergency` | 突发情况应急处置步骤（JSON） |

## 目录结构

```text
son_of_sea/
├── app/
│   ├── __init__.py
│   ├── deepseek_client.py      # DeepSeek 客户端（流式 + 非流式）
│   ├── retrieval.py            # RAG 检索模块（ChromaDB）
│   └── init_knowledge_base.py  # 知识库初始化脚本
├── data/
│   ├── knowledge_base.json     # 知识库（47 条建筑安全规范）
│   └── chroma_db/              # 向量库（自动生成，已忽略提交）
├── static/
│   └── index.html              # 单文件前端
├── main.py                     # FastAPI 入口（4 个接口 + 日志中间件）
├── requirements.txt
├── .env.example
└── README.md
```

## 本地运行

```powershell
cd "E:\AI agent\son_of_sea"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 复制 .env.example 为 .env 并填入 DEEPSEEK_API_KEY
cp .env.example .env

# 初始化知识库（首次运行会自动下载嵌入模型，约 90MB）
python -m app.init_knowledge_base

# 启动服务
uvicorn main:app --reload
```

- 前端：<http://127.0.0.1:8000>
- 接口文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

## 公网运行（Cloud Studio / 云服务器）

公网环境必须监听 `0.0.0.0`，不能使用仅本机可访问的 `127.0.0.1`：

```bash
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8001
```

项目附带 `Dockerfile` 和 `.devcontainer/devcontainer.json`，可用于 Cloud Studio
导入后的自动安装和 8001 端口转发。请在 Cloud Studio 的环境变量/密钥设置中配置
`DEEPSEEK_API_KEY`，不要上传 `.env`。Cloud Studio 预览地址适合比赛演示，长期公网服务
建议使用腾讯云轻量应用服务器并在 Nginx 后配置 HTTPS。

> 说明：启动时若检测到向量库未初始化，会自动从 `data/knowledge_base.json` 入库，无需手动执行初始化脚本。

## 知识库与 RAG

- 知识库 `data/knowledge_base.json` 包含 47 条结构化规范数据，覆盖：高空作业、临时用电、起重吊装、消防安全、工种操作规范（钢筋/木/电/架子/混凝土等）、应急处置流程、个人防护装备、通用安全管理。
- 每条含 `category`、`title`、`content`、`keywords` 四个字段。
- 检索使用 ChromaDB 持久化存储 + `sentence-transformers/all-MiniLM-L6-v2` 嵌入模型，支持按分类过滤检索。
- 检索结果（top 5）注入系统提示词后调用 DeepSeek，回答基于本地知识库，降低幻觉风险。

## API 接口

### 1. 安全问答 `POST /chat`（SSE 流式）

```json
{
  "message": "高处作业前要检查什么？",
  "history": [{"role": "user", "content": "我在做外墙施工。"}]
}
```

响应：

```text
data: {"type":"delta","content":"首先..."}
data: {"type":"done"}
```

### 2. 隐患识别 `POST /analyze_hazard`

```json
{ "description": "3楼外墙施工，脚手架没有安全网，工人没系安全带" }
```

响应：

```json
{
  "risks": [
    { "category": "高处作业防护缺失", "description": "...", "risk_level": "高", "suggestion": "..." }
  ],
  "summary": "总体评估",
  "disclaimer": "以上分析仅供参考..."
}
```

### 3. 技能微课 `POST /generate_lesson`

```json
{ "trade": "钢筋工" }
```

支持工种：钢筋工、木工、电工、架子工、混凝土工、焊工、塔吊司机。

响应：

```json
{
  "trade": "钢筋工",
  "intro": "工种简介",
  "core_skills": ["技能1", "..."],
  "steps": [{"step": 1, "title": "步骤名", "detail": "操作说明"}],
  "safety_notes": ["注意事项"],
  "common_mistakes": ["常见错误"],
  "disclaimer": "以现场交底为准"
}
```

### 4. 应急指南 `POST /emergency`

```json
{ "situation": "有人中暑了" }
```

响应：

```json
{
  "situation": "情况识别",
  "immediate_actions": [{"order": 1, "action": "立即行动", "detail": "操作要点"}],
  "dont_do": ["禁忌事项"],
  "when_to_call": {"call_120": true, "call_119": false, "reason": "..."},
  "disclaimer": "以现场安全员和医疗专业人员指导为准"
}
```

## 环境变量（.env）

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/anthropic
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_API_VERSION=2023-06-01
DEEPSEEK_MAX_TOKENS=2048
DEEPSEEK_TIMEOUT=120
```

## 安全说明

- `.env` 和 `data/chroma_db/` 已被 `.gitignore` 忽略，真实 API Key 不会提交到仓库。
- AI 回答仅供辅助参考，不能替代现场安全管理制度、持证专业人员和应急救援机构。
- 应急指南接口在模型调用失败时返回兜底基础建议，保证可用性。

## 比赛提交材料

`submission/` 目录包含可编辑的参赛材料草稿：作品说明文档、2 分钟视频脚本、
人机协同履历表和提交使用说明。案例放在 `data/demo_scenarios.json`，全部明确标注为模拟演示，
不代表真实工地部署记录。正式打包时请排除 `.env`、`.venv`、日志和 `data/chroma_db/`。
