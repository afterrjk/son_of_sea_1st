"""DeepSeek Anthropic 兼容接口的异步调用封装。

这个模块只关心模型通信，不包含 FastAPI 或页面逻辑，后续可以在命令行、
定时任务以及 RAG 检索流程中复用。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx


class DeepSeekAPIError(RuntimeError):
    """DeepSeek 接口返回错误或返回了无法处理的数据。"""


class DeepSeekClient:
    """调用 DeepSeek Anthropic 兼容 Messages API 的轻量客户端。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com/anthropic",
        model: str = "deepseek-v4-flash",
        api_version: str = "2023-06-01",
        timeout: float = 120.0,
    ) -> None:
        if not api_key:
            raise ValueError("未配置 DEEPSEEK_API_KEY")

        self.model = model
        self._messages_url = f"{base_url.rstrip('/')}/v1/messages"
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=20.0),
            headers={
                "x-api-key": api_key,
                "anthropic-version": api_version,
                "content-type": "application/json",
                "accept": "text/event-stream",
            },
        )

    async def close(self) -> None:
        """释放连接池；应用退出时应调用此方法。"""

        await self._http.aclose()

    async def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        system_prompt: str,
        max_tokens: int = 2048,
    ) -> str:
        """发送非流式对话请求，返回完整的模型回复文本。

        适用于需要结构化 JSON 返回或不需逐字展示的场景。
        """
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": list(messages),
            "stream": False,
        }

        try:
            response = await self._http.post(
                self._messages_url, json=payload
            )
            if response.is_error:
                error_body = response.text[:1000]
                raise DeepSeekAPIError(
                    f"DeepSeek API 请求失败（HTTP {response.status_code}）："
                    f"{error_body}"
                )
            data = response.json()
            # Anthropic Messages API 非流式响应格式：
            # {"id": "...", "content": [{"type": "text", "text": "..."}], ...}
            content_blocks = data.get("content", [])
            texts: list[str] = []
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(str(block.get("text", "")))
            return "\n".join(texts)
        except httpx.TimeoutException as exc:
            raise DeepSeekAPIError("连接 DeepSeek API 超时，请稍后重试") from exc
        except httpx.RequestError as exc:
            raise DeepSeekAPIError(f"无法连接 DeepSeek API：{exc}") from exc

    async def stream_chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        system_prompt: str,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """发送多轮对话并逐段产出模型文本。

        ``messages`` 遵循 Anthropic Messages API 格式，例如：
        ``[{"role": "user", "content": "安全帽如何正确佩戴？"}]``。
        方法会解析上游 SSE，只向调用方产出 text_delta 中的新增文字。
        """

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": list(messages),
            "stream": True,
        }

        try:
            async with self._http.stream(
                "POST", self._messages_url, json=payload
            ) as response:
                if response.is_error:
                    error_body = (await response.aread()).decode(
                        response.encoding or "utf-8", errors="replace"
                    )
                    raise DeepSeekAPIError(
                        f"DeepSeek API 请求失败（HTTP {response.status_code}）："
                        f"{error_body[:1000]}"
                    )

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue

                    raw_data = line.removeprefix("data:").strip()
                    if not raw_data or raw_data == "[DONE]":
                        continue

                    try:
                        event = json.loads(raw_data)
                    except json.JSONDecodeError:
                        continue

                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            yield str(delta["text"])
                    elif event.get("type") == "error":
                        error = event.get("error", {})
                        message = error.get("message", "模型流返回未知错误")
                        raise DeepSeekAPIError(str(message))
        except httpx.TimeoutException as exc:
            raise DeepSeekAPIError("连接 DeepSeek API 超时，请稍后重试") from exc
        except httpx.RequestError as exc:
            raise DeepSeekAPIError(f"无法连接 DeepSeek API：{exc}") from exc

