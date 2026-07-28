"""“筑安”FastAPI 应用入口。

开发环境启动命令：uvicorn main:app --reload
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.deepseek_client import DeepSeekAPIError, DeepSeekClient
from app.retrieval import search_knowledge, format_retrieval_context

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("zhuan")

# ---------------------------------------------------------------------------
# 提示词模板
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """你是“筑安”，一名服务建筑工地一线产业工人的安全生产助手。
你的回答必须准确、清楚、可执行，优先使用简短步骤和通俗中文。
涉及高风险作业、事故或人身伤害时，必须提醒用户立即停止危险操作、撤离或求助现场安全员，
并明确说明AI建议不能替代施工方案、现场安全管理制度、持证专业人员和应急救援机构。
信息不足时先询问关键现场条件，不要编造规范条文、标准编号或救援电话。"""

RAG_SYSTEM_PROMPT_TEMPLATE = """你是一个专业的建筑工地安全与技能助手，名为“筑安”。
你的职责是帮助建筑工人解答安全问题、提供技能指导、处理突发情况。
请基于以下参考知识回答用户的问题。如果参考知识不足以回答，请明确告知用户并建议咨询现场安全员。
参考知识：
{retrieved_knowledge}

用户问题：{user_query}

要求：
- 回答要通俗易懂，用工人能理解的语言
- 涉及安全问题时，要强调“最终以现场安全员和专业规范为准”
- 回答要结构化，使用序号和分段"""

HAZARD_ANALYSIS_PROMPT = """你是一名建筑工地安全隐患识别专家。请根据以下参考知识和用户描述，分析现场存在的安全隐患。

参考知识：
{retrieved_knowledge}

用户对现场情况的描述：
{description}

请按以下格式输出分析结果（仅输出 JSON，不要有任何额外文字）：
{{
  "risks": [
    {{
      "category": "隐患类别",
      "description": "具体隐患描述",
      "risk_level": "高/中/低",
      "suggestion": "处置建议"
    }}
  ],
  "summary": "总体安全评估",
  "disclaimer": "以上分析仅供参考，最终以现场安全员和专业规范为准。发现重大隐患应立即停止作业并报告现场负责人。"
}}"""

LESSON_GENERATION_PROMPT = """你是一名建筑工地技能培训师。请根据以下参考知识，为{trade}工种生成一份结构化的技能微课。

参考知识：
{retrieved_knowledge}

请按以下格式输出微课内容（仅输出 JSON，不要有任何额外文字）：
{{
  "trade": "{trade}",
  "intro": "工种简介（2-3句话）",
  "core_skills": ["核心技能点1", "核心技能点2", "核心技能点3", "核心技能点4", "核心技能点5"],
  "steps": [
    {{"step": 1, "title": "步骤名称", "detail": "具体操作说明"}}
  ],
  "safety_notes": ["安全注意事项1", "安全注意事项2", "安全注意事项3"],
  "common_mistakes": ["常见错误1", "常见错误2", "常见错误3"],
  "disclaimer": "以上内容基于通用规范，具体操作以现场施工方案和安全技术交底为准。"
}}"""

EMERGENCY_RESPONSE_PROMPT = """你是一名建筑工地应急救援指导员。请根据以下参考知识，针对用户描述的紧急情况给出应急处置指导。

参考知识：
{retrieved_knowledge}

紧急情况描述：
{situation}

请按以下格式输出应急指南（仅输出 JSON，不要有任何额外文字）：
{{
  "situation": "情况识别",
  "immediate_actions": [
    {{"order": 1, "action": "具体行动步骤", "detail": "操作要点"}}
  ],
  "dont_do": ["禁忌事项1", "禁忌事项2", "禁忌事项3"],
  "when_to_call": {{
    "call_120": true/false,
    "call_119": true/false,
    "reason": "需要呼叫的原因"
  }},
  "disclaimer": "本指南仅供紧急参考，必须以现场安全员和医疗专业人员指导为准。"
}}"""

# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=30)
    system_prompt: str | None = Field(default=None, max_length=20_000)


class HazardRequest(BaseModel):
    description: str = Field(
        min_length=1, max_length=5_000,
        description="用户对现场情况的文字描述",
    )


class LessonRequest(BaseModel):
    trade: str = Field(
        min_length=1, max_length=20,
        description="工种名称：钢筋工、木工、电工、架子工、混凝土工",
    )


class EmergencyRequest(BaseModel):
    situation: str = Field(
        min_length=1, max_length=2_000,
        description="紧急情况描述，如'有人中暑了'",
    )


VALID_TRADES = {"钢筋工", "木工", "电工", "架子工", "混凝土工", "焊工", "塔吊司机"}

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def create_client() -> DeepSeekClient:
    return DeepSeekClient(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/anthropic"),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        api_version=os.getenv("DEEPSEEK_API_VERSION", "2023-06-01"),
        timeout=float(os.getenv("DEEPSEEK_TIMEOUT", "120")),
    )


# 模块级客户端缓存，避免每次请求都重建 HTTP 连接池
_client: DeepSeekClient | None = None


def get_client() -> DeepSeekClient:
    """获取或创建 DeepSeek 客户端实例（模块级单例）。"""
    global _client
    if _client is None:
        _client = create_client()
    return _client


async def _call_deepseek_nonstream(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 2048,
) -> str:
    """调用 DeepSeek 非流式接口，返回完整文本。"""
    client = get_client()
    return await client.chat(
        [{"role": "user", "content": user_message}],
        system_prompt=system_prompt,
        max_tokens=max_tokens,
    )


def _parse_json_response(text: str) -> dict:
    """从模型返回文本中提取 JSON 对象。"""
    text = text.strip()
    # 尝试去除可能的 markdown 代码块包裹
    if text.startswith("```"):
        lines = text.split("\n")
        # 去除首行 ```json 或 ``` 和末行 ```
        if len(lines) > 2:
            text = "\n".join(lines[1:-1])
        else:
            text = text.strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"模型返回非 JSON 文本，原样返回。前200字符：{text[:200]}")
        return {"raw_response": text}


def sse_event(event_type: str, **data: object) -> str:
    payload = {"type": event_type, **data}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

# ---------------------------------------------------------------------------
# 应用 & 生命周期
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化知识库
    try:
        from app.retrieval import get_retriever
        retriever = get_retriever()
        if not retriever.is_initialized():
            count = retriever.init_from_json()
            logger.info(f"知识库自动初始化完成，入库 {count} 条。")
        else:
            logger.info(f"知识库已就绪，共 {retriever.collection.count()} 条。")
    except Exception as exc:
        logger.warning(f"知识库初始化失败：{exc}，RAG 功能将不可用。")
    yield
    global _client
    if _client is not None:
        await _client.close()
        _client = None

app = FastAPI(
    title="筑安 AI 智能体",
    description="面向建筑工地一线产业工人的安全生产智能助手",
    version="0.2.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# 请求日志中间件
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} "
        f"({duration:.3f}s)"
    )
    return response

# ---------------------------------------------------------------------------
# 静态页面 & 健康检查
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

# ===================================================================
# 接口一：安全问答 /chat（SSE 流式）
# ===================================================================

@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """接收当前问题和历史消息，RAG 检索知识库后以 SSE 流式返回模型回答。"""

    async def event_stream() -> AsyncIterator[str]:
        try:
            client = get_client()

            if request.system_prompt:
                system_prompt = request.system_prompt
            else:
                retrieved = search_knowledge(request.message, top_k=5)
                context = format_retrieval_context(retrieved)
                system_prompt = RAG_SYSTEM_PROMPT_TEMPLATE.format(
                    retrieved_knowledge=context,
                    user_query=request.message,
                )

            messages = [item.model_dump() for item in request.history]
            messages.append({"role": "user", "content": request.message})

            async for text in client.stream_chat(
                messages,
                system_prompt=system_prompt,
                max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "2048")),
            ):
                yield sse_event("delta", content=text)
            yield sse_event("done")
        except (DeepSeekAPIError, ValueError) as exc:
            logger.error(f"/chat 错误：{exc}")
            yield sse_event("error", message=str(exc))
        except Exception as exc:
            logger.exception(f"/chat 未知错误")
            yield sse_event("error", message="服务暂时不可用，请稍后重试")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# ===================================================================
# 接口二：隐患识别辅助 /analyze_hazard
# ===================================================================

@app.post("/analyze_hazard")
async def analyze_hazard(request: HazardRequest) -> JSONResponse:
    """识别现场安全隐患，返回结构化的风险评估结果。

    处理流程：
    1. 从知识库检索安全隐患相关内容
    2. 调用 DeepSeek API 分析隐患、风险等级和处置建议
    3. 返回 JSON 结构化结果
    """
    try:
        # 1. RAG 检索
        retrieved = search_knowledge(request.description, top_k=5)
        context = format_retrieval_context(retrieved)

        # 2. 构建提示词并调用模型
        system_prompt = HAZARD_ANALYSIS_PROMPT.format(
            retrieved_knowledge=context,
            description=request.description,
        )

        response = await _call_deepseek_nonstream(
            system_prompt=system_prompt,
            user_message=request.description,
            max_tokens=2048,
        )

        # 3. 解析 JSON
        result = _parse_json_response(response)
        return JSONResponse(content=result)

    except DeepSeekAPIError as exc:
        logger.error(f"/analyze_hazard DeepSeek 错误：{exc}")
        return JSONResponse(
            status_code=502,
            content={"error": str(exc), "disclaimer": "AI分析失败，请咨询现场安全员。"},
        )
    except Exception as exc:
        logger.exception(f"/analyze_hazard 未知错误")
        return JSONResponse(
            status_code=500,
            content={"error": "服务暂时不可用，请稍后重试"},
        )

# ===================================================================
# 接口三：技能微课生成 /generate_lesson
# ===================================================================

@app.post("/generate_lesson")
async def generate_lesson(request: LessonRequest) -> JSONResponse:
    """根据工种生成结构化技能微课。

    支持工种：钢筋工、木工、电工、架子工、混凝土工、焊工、塔吊司机
    """
    trade = request.trade.strip()

    if trade not in VALID_TRADES:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"不支持的工种「{trade}」，可选：{'、'.join(sorted(VALID_TRADES))}",
            },
        )

    try:
        # 1. RAG 检索：优先搜索对应工种的操作规范
        category_map = {
            "钢筋工": "钢筋工操作规范",
            "木工": "木工操作规范",
            "电工": "电工操作规范",
            "架子工": "架子工操作规范",
            "混凝土工": "混凝土工操作规范",
            "焊工": "钢筋工操作规范",       # 焊工规范部分揉在钢筋工里
            "塔吊司机": "起重吊装安全规范",  # 塔吊司机参考起重吊装
        }
        category = category_map.get(trade, None)

        # 先用分类过滤，如果命中少则追加通用搜索
        if category:
            results = search_knowledge(
                f"{trade} 操作规范 安全操作规程", top_k=5, category_filter=category
            )
        else:
            results = []

        # 如果分类检索结果不够，补充通用检索
        if len(results) < 3:
            extra = search_knowledge(
                f"{trade} 安全操作 技能培训 施工规范", top_k=5
            )
            seen = {r["id"] for r in results}
            for r in extra:
                if r["id"] not in seen:
                    results.append(r)
                    seen.add(r["id"])
                    if len(results) >= 5:
                        break

        context = format_retrieval_context(results)

        # 2. 构建提示词并调用模型
        system_prompt = LESSON_GENERATION_PROMPT.format(
            trade=trade,
            retrieved_knowledge=context,
        )

        response = await _call_deepseek_nonstream(
            system_prompt=system_prompt,
            user_message=f"请为{trade}工种生成技能微课",
            max_tokens=2048,
        )

        # 3. 解析 JSON
        result = _parse_json_response(response)
        return JSONResponse(content=result)

    except DeepSeekAPIError as exc:
        logger.error(f"/generate_lesson DeepSeek 错误：{exc}")
        return JSONResponse(
            status_code=502,
            content={"error": str(exc), "trade": trade},
        )
    except Exception as exc:
        logger.exception(f"/generate_lesson 未知错误")
        return JSONResponse(
            status_code=500,
            content={"error": "服务暂时不可用，请稍后重试", "trade": trade},
        )

# ===================================================================
# 接口四：应急指南 /emergency
# ===================================================================

@app.post("/emergency")
async def emergency(request: EmergencyRequest) -> JSONResponse:
    """根据紧急情况描述生成应急处置指南。

    支持场景：中暑、坠落、触电、火灾、坍塌、物体打击、化学品伤害等。
    """
    try:
        # 1. RAG 检索：精确检索应急处置流程
        results = search_knowledge(
            request.situation, top_k=5, category_filter="应急处置流程"
        )

        # 如果应急处置分类命中不够，放宽到全局
        if len(results) < 2:
            extra = search_knowledge(request.situation, top_k=5)
            seen = {r["id"] for r in results}
            for r in extra:
                if r["id"] not in seen:
                    results.append(r)
                    seen.add(r["id"])
                    if len(results) >= 5:
                        break

        context = format_retrieval_context(results)

        # 2. 构建提示词并调用模型
        system_prompt = EMERGENCY_RESPONSE_PROMPT.format(
            retrieved_knowledge=context,
            situation=request.situation,
        )

        response = await _call_deepseek_nonstream(
            system_prompt=system_prompt,
            user_message=request.situation,
            max_tokens=2048,
        )

        # 3. 解析 JSON
        result = _parse_json_response(response)
        return JSONResponse(content=result)

    except DeepSeekAPIError as exc:
        logger.error(f"/emergency DeepSeek 错误：{exc}")
        return JSONResponse(
            status_code=502,
            content={
                "error": str(exc),
                "immediate_actions": [{"order": 1, "action": "保持冷静", "detail": "立即拨打120急救电话并报告现场安全员。"}],
                "dont_do": ["不要盲目施救", "不要移动伤者（除非现场有立即危险）"],
                "when_to_call": {"call_120": True, "call_119": False, "reason": "有人员伤亡风险"},
                "disclaimer": "AI服务异常，以上为基础建议，请以现场安全员和医疗专业人员指导为准。",
            },
        )
    except Exception as exc:
        logger.exception(f"/emergency 未知错误")
        return JSONResponse(
            status_code=500,
            content={"error": "服务暂时不可用，请稍后重试"},
        )
