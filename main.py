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

RIGHTS_PROMPT = """你是"筑安"劳动权益助手，服务建筑工地一线产业工人。
工友可能会咨询工资拖欠、工伤保险、劳动合同、加班费、社保缴纳、欠薪维权、劳务纠纷等问题。
回答要求：
1. 用通俗中文，避免生硬法律术语，必要时用大白话解释；
2. 先讲清"是什么/是否合法"，再给"怎么做"的具体步骤；
3. 给出维权渠道时尽量具体：如项目部/劳务公司沟通、项目所在地劳动保障监察大队、12333人社热线、12345政务服务热线、工会等；
4. 提醒保留证据：劳动合同、考勤记录、工资条、微信聊天记录、转账记录等；
5. 结尾注明"以上为一般性信息参考，不构成法律意见，重大纠纷建议咨询专业律师或当地工会/劳动监察部门"。
6. 信息不足时，先询问关键情况（如是否签合同、拖欠多久、金额多少），不要臆测。

用户咨询内容：
{question}

请以结构清晰、分点的方式回答。"""

COMFORT_PROMPT = """你是"筑安"工地暖心陪伴助手，一个温暖、真诚、耐心的倾听者。
建筑工人群体常面临离家思乡、工作疲惫、压力大、孤独、担忧家庭、人际关系困扰、情绪低落等情况。
回答要求：
1. 先共情、理解和接纳对方的情绪，语气温暖亲切，不说教、不评判；
2. 用1-2个问题或回应让对方感到被看见、被尊重；
3. 提供一些简单可操作的情绪调节小方法（如深呼吸、和工友聊聊、给家里打个电话、记录三件小确幸等）；
4. 如果需要，可以鼓励对方向身边信任的人倾诉，或联系项目工会、心理援助热线（如12356全国心理援助热线）；
5. 如果对方表达出自伤、自残或伤害他人的危险想法，必须认真对待，明确建议立即联系专业人员或拨打120/110，并告知这是紧急情况；
6. 保持适度篇幅，真诚自然，不要过于机械或模板化。

工友的倾诉：
{message}

请以温暖的口吻回应。"""

LOGISTICS_PROMPT = """你是"筑安"工地生活服务助手，为一线产业工人提供工地后勤生活保障信息。
可以解答的问题包括：宿舍住宿、食堂用餐、通勤班车、工装劳保用品领取、体检安排、防暑防寒保障、
饮水休息点、医疗点位置、证件办理（如实名制入场、银行卡）、手机充电与网络、节假日安排等。
回答要求：
1. 用通俗中文，简洁清楚，先说结论再给细节；
2. 具体地点、时间、流程等项目部有明确规定时，提示"以项目部/班组通知为准"；
3. 给出"找不到人/不清楚找谁"时的兜底建议（如找班组长、项目综合办、安全员）；
4. 涉及健康或安全的事项（如中暑、受伤）要提示及时就医和报告；
5. 信息不足时，先询问是哪个环节/哪类生活需求，再给出通用指引。

工友的问题：
{question}

请以分点、清晰的方式回答。"""

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


class RightsRequest(BaseModel):
    question: str = Field(
        min_length=1, max_length=2_000,
        description="劳动权益咨询问题，如'工资被拖欠两个月怎么办'",
    )


class ComfortRequest(BaseModel):
    message: str = Field(
        min_length=1, max_length=2_000,
        description="工友倾诉或情绪表达的内容",
    )


class LogisticsRequest(BaseModel):
    question: str = Field(
        min_length=1, max_length=2_000,
        description="工地后勤生活问题，如'宿舍冬天太冷怎么办'",
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
    # 健康检查不触发模型请求，但会告诉演示人员本地 RAG 是否已初始化。
    rag_status = "ready"
    try:
        retriever = __import__("app.retrieval", fromlist=["get_retriever"]).get_retriever()
        rag_status = "ready" if retriever.is_initialized() else "not_initialized"
    except Exception:
        rag_status = "unavailable"
    return {"status": "ok", "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"), "rag": rag_status}

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
            status_code=200,
            content={
                "fallback": True,
                "error": str(exc),
                "risks": [{
                    "category": "需要现场复核",
                    "description": "AI服务暂时不可用，无法完成自动风险分级。",
                    "risk_level": "中",
                    "suggestion": "立即停止可能造成伤害的作业，设置警戒并报告现场安全员逐项检查。",
                }],
                "summary": "这是基础安全兜底提示，不代表对现场风险的最终判断。",
                "disclaimer": "AI服务异常。请以现场安全员、施工方案和现行规范为准。",
            },
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
            status_code=200,
            content={
                "fallback": True,
                "error": str(exc),
                "trade": trade,
                "intro": "AI服务暂时不可用，先提供一份班前复习提纲。",
                "core_skills": ["作业前检查工具和防护用品", "按交底要求进行操作", "发现异常立即停工报告"],
                "steps": [
                    {"step": 1, "title": "班前检查", "detail": "确认作业面、工具、临边洞口和个人防护用品状态。"},
                    {"step": 2, "title": "按方案作业", "detail": "严格执行施工方案和安全技术交底，不擅自改变工艺。"},
                    {"step": 3, "title": "收工复查", "detail": "清理现场，切断设备电源，向班组长报告异常。"},
                ],
                "safety_notes": ["严禁违章指挥、违章作业和冒险作业", "不清楚时先停下来询问现场安全员"],
                "common_mistakes": ["未检查工具就开工", "为赶进度省略防护措施"],
                "disclaimer": "AI服务异常。以上为基础复习提纲，必须以现场交底和专业规范为准。",
            },
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

# ===================================================================
# 接口五：劳动权益咨询 /rights
# ===================================================================

@app.post("/rights")
async def rights(request: RightsRequest) -> JSONResponse:
    """解答工友劳动权益问题：工资、合同、工伤、社保、维权渠道等。"""
    try:
        # 权益类问题主要依赖常识与政策口径，不强依赖知识库检索
        system_prompt = RIGHTS_PROMPT.format(question=request.question)
        response = await _call_deepseek_nonstream(
            system_prompt=system_prompt,
            user_message=request.question,
            max_tokens=2048,
        )
        return JSONResponse(content={"answer": response})

    except DeepSeekAPIError as exc:
        logger.error(f"/rights DeepSeek 错误：{exc}")
        return JSONResponse(
            status_code=200,
            content={
                "fallback": True,
                "error": str(exc),
                "answer": (
                    "AI服务暂时不可用，请先参考以下通用步骤：\n\n"
                    "1. 整理证据：劳动合同、考勤记录、工资条、微信/银行转账记录等；\n"
                    "2. 先与班组长、劳务公司或项目部沟通核实；\n"
                    "3. 无法解决时拨打 **12333**（人社热线）或 **12345**（政务服务热线）反映；\n"
                    "4. 向项目所在地的劳动保障监察大队投诉，或联系工会组织；\n"
                    "5. 涉及工伤，务必保留就医记录并及时申请工伤认定。\n\n"
                    "以上为一般性信息参考，不构成法律意见，建议咨询专业律师或当地劳动监察部门。"
                ),
            },
        )
    except Exception as exc:
        logger.exception(f"/rights 未知错误")
        return JSONResponse(
            status_code=500,
            content={"error": "服务暂时不可用，请稍后重试"},
        )

# ===================================================================
# 接口六：心理陪伴 /comfort
# ===================================================================

@app.post("/comfort")
async def comfort(request: ComfortRequest) -> JSONResponse:
    """为工友提供温暖陪伴与情绪疏导。"""
    try:
        system_prompt = COMFORT_PROMPT.format(message=request.message)
        response = await _call_deepseek_nonstream(
            system_prompt=system_prompt,
            user_message=request.message,
            max_tokens=1024,
        )
        return JSONResponse(content={"reply": response})

    except DeepSeekAPIError as exc:
        logger.error(f"/comfort DeepSeek 错误：{exc}")
        return JSONResponse(
            status_code=200,
            content={
                "fallback": True,
                "error": str(exc),
                "reply": (
                    "谢谢你愿意和我分享。出门在外打工不容易，想家、累、委屈都是正常的，\n"
                    "你的感受值得被认真对待。可以先停下来喝口水、深呼吸几次，\n"
                    "如果方便，给家人打个电话，或者和信得过的工友聊聊。\n\n"
                    "如果觉得情绪一直压着，也可以联系项目工会，或拨打 **12356** 心理援助热线，\n"
                    "那里有专业人员愿意倾听。你并不孤单。"
                ),
            },
        )
    except Exception as exc:
        logger.exception(f"/comfort 未知错误")
        return JSONResponse(
            status_code=500,
            content={"error": "服务暂时不可用，请稍后重试"},
        )

# ===================================================================
# 接口七：后勤保障 /logistics
# ===================================================================

@app.post("/logistics")
async def logistics(request: LogisticsRequest) -> JSONResponse:
    """解答工地后勤生活保障问题：住宿、食堂、通勤、体检等。"""
    try:
        system_prompt = LOGISTICS_PROMPT.format(question=request.question)
        response = await _call_deepseek_nonstream(
            system_prompt=system_prompt,
            user_message=request.question,
            max_tokens=1536,
        )
        return JSONResponse(content={"answer": response})

    except DeepSeekAPIError as exc:
        logger.error(f"/logistics DeepSeek 错误：{exc}")
        return JSONResponse(
            status_code=200,
            content={
                "fallback": True,
                "error": str(exc),
                "answer": (
                    "AI服务暂时不可用，请先参考以下通用指引：\n\n"
                    "1. **住宿问题**：先向班组长或项目综合办公室反映，说明具体问题（如暖气、漏水、卫生）；\n"
                    "2. **食堂问题**：对饭菜质量或价格有意见，可向项目部后勤负责人反馈，或通过职工代表/工会提出；\n"
                    "3. **通勤班车**：班次与路线以项目部通知为准，具体可问班组长；\n"
                    "4. **劳保用品**：按规定领用安全帽、工装等，不足时向班组长申领；\n"
                    "5. **身体不适**：及时到项目医务室/附近医院就诊，并向班组长报备。\n\n"
                    "具体安排以项目部通知为准。"
                ),
            },
        )
    except Exception as exc:
        logger.exception(f"/logistics 未知错误")
        return JSONResponse(
            status_code=500,
            content={"error": "服务暂时不可用，请稍后重试"},
        )
