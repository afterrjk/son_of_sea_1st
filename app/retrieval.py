"""知识库检索模块 —— 基于 ChromaDB + sentence-transformers 的本地 RAG 检索。

提供：
- init_knowledge_base()：从 JSON 文件加载知识条目并向量化存入 ChromaDB
- search()：根据用户查询检索最相关的知识条目
- get_retrieval_context()：将检索结果格式化为可直接注入系统提示词的文本
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypedDict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "zhuan_safety_knowledge"


class RetrievalResult(TypedDict):
    """单条检索结果。"""

    id: str
    category: str
    title: str
    content: str
    keywords: list[str]
    score: float


class KnowledgeRetriever:
    """ChromaDB 知识库检索器，封装向量化、存储和查询功能。"""

    def __init__(self) -> None:
        self._embedding_model: SentenceTransformer | None = None
        self._chroma_client: chromadb.PersistentClient | None = None
        self._collection: chromadb.Collection | None = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        """延迟加载向量模型，避免 import 时下载模型。"""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self._embedding_model

    @property
    def chroma_client(self) -> chromadb.PersistentClient:
        if self._chroma_client is None:
            os.makedirs(str(CHROMA_DIR), exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
        return self._chroma_client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "筑安建筑安全知识库"},
            )
        return self._collection

    def is_initialized(self) -> bool:
        """检查知识库是否已经完成向量化入库。"""
        try:
            return self.collection.count() > 0
        except Exception:
            return False

    def init_from_json(self, json_path: str | None = None) -> int:
        """从 JSON 文件加载知识条目，逐条向量化并存入 ChromaDB。

        Args:
            json_path: 知识库 JSON 文件路径，默认使用 data/knowledge_base.json

        Returns:
            已入库的条目数量
        """
        if json_path is None:
            json_path = str(DATA_DIR / "knowledge_base.json")

        with open(json_path, "r", encoding="utf-8") as f:
            entries: list[dict] = json.load(f)

        if not entries:
            return 0

        ids = [entry["id"] for entry in entries]

        # 构造用于向量化的文本：标题 + 关键词 + 正文（截断以控制嵌入质量）
        texts: list[str] = []
        for entry in entries:
            kw_str = "，".join(entry.get("keywords", []))
            content = entry.get("content", "")
            texts.append(
                f"【{entry['category']}】{entry['title']}\n"
                f"关键词：{kw_str}\n"
                f"{content[:1500]}"
            )

        # 生成向量并批量入库
        embeddings = self.embedding_model.encode(
            texts, show_progress_bar=True, convert_to_numpy=True
        ).tolist()

        metadatas = [
            {
                "category": entry["category"],
                "title": entry["title"],
                "keywords": ", ".join(entry.get("keywords", [])),
            }
            for entry in entries
        ]

        documents = [entry["content"] for entry in entries]

        # 先清空旧数据再写入
        try:
            self.chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self._collection = self.chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "筑安建筑安全知识库"},
        )

        batch_size = 10
        total = len(entries)
        for i in range(0, total, batch_size):
            batch_slice = slice(i, i + batch_size)
            self.collection.add(
                ids=ids[batch_slice],
                embeddings=embeddings[batch_slice],
                metadatas=metadatas[batch_slice],
                documents=documents[batch_slice],
            )

        return total

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """根据用户问题检索最相关的知识条目。

        Args:
            query: 用户输入的问题文本
            top_k: 返回的最相关条目数量

        Returns:
            检索结果列表，按相似度降序排列，每条含 id/category/title/content/keywords/score
        """
        if not self.is_initialized():
            return []

        query_embedding = self.embedding_model.encode(
            [query], show_progress_bar=False, convert_to_numpy=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output: list[RetrievalResult] = []
        if not results["ids"] or not results["ids"][0]:
            return output

        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            document = results["documents"][0][i] if results["documents"] else ""
            distance = results["distances"][0][i] if results["distances"] else 0.0

            # ChromaDB 返回的是余弦距离，转换为相似度分数 (0~1)
            # cosine_similarity = 1 - cosine_distance
            similarity = max(0.0, min(1.0, 1.0 - distance))

            keywords_raw = metadata.get("keywords", "")
            keywords = [kw.strip() for kw in keywords_raw.split(",") if kw.strip()] if keywords_raw else []

            output.append(
                {
                    "id": doc_id,
                    "category": metadata.get("category", ""),
                    "title": metadata.get("title", ""),
                    "content": document,
                    "keywords": keywords,
                    "score": round(similarity, 4),
                }
            )

        return output

    def format_context(self, results: list[RetrievalResult]) -> str:
        """将检索结果格式化为可直接注入系统提示词的文本块。

        Args:
            results: search() 返回的检索结果列表

        Returns:
            格式化的参考知识文本
        """
        if not results:
            return "暂无相关参考知识。"

        parts: list[str] = []
        for idx, r in enumerate(results, 1):
            parts.append(
                f"【参考 {idx}】{r['category']} — {r['title']}\n"
                f"{r['content']}\n"
            )
        return "\n".join(parts)


# 模块级单例，供 main.py 直接导入使用
_retriever: KnowledgeRetriever | None = None


def get_retriever() -> KnowledgeRetriever:
    """获取全局检索器单例。"""
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever()
    return _retriever


def search_knowledge(
    query: str, top_k: int = 5, category_filter: str | None = None
) -> list[RetrievalResult]:
    """便捷函数：检索知识库。

    如果知识库尚未初始化，则自动从 knowledge_base.json 加载并入库。
    支持按 category 字段过滤（如 "应急处置流程"、"钢筋工操作规范"）。
    """
    retriever = get_retriever()
    if not retriever.is_initialized():
        json_path = DATA_DIR / "knowledge_base.json"
        if json_path.exists():
            count = retriever.init_from_json(str(json_path))
            print(f"[RAG] 知识库自动初始化完成，入库 {count} 条。")
        else:
            return []

    if not retriever.is_initialized():
        return []

    # 构建查询向量
    query_embedding = retriever.embedding_model.encode(
        [query], show_progress_bar=False, convert_to_numpy=True
    ).tolist()

    # 构建 ChromaDB where 过滤条件
    where_filter = None
    if category_filter:
        where_filter = {"category": category_filter}

    results = retriever.collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, retriever.collection.count()),
        include=["documents", "metadatas", "distances"],
        where=where_filter,
    )

    output: list[RetrievalResult] = []
    if not results["ids"] or not results["ids"][0]:
        return output

    for i, doc_id in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][i] if results["metadatas"] else {}
        document = results["documents"][0][i] if results["documents"] else ""
        distance = results["distances"][0][i] if results["distances"] else 0.0
        similarity = max(0.0, min(1.0, 1.0 - distance))
        keywords_raw = metadata.get("keywords", "")
        keywords = [kw.strip() for kw in keywords_raw.split(",") if kw.strip()] if keywords_raw else []

        output.append({
            "id": doc_id,
            "category": metadata.get("category", ""),
            "title": metadata.get("title", ""),
            "content": document,
            "keywords": keywords,
            "score": round(similarity, 4),
        })

    return output


def format_retrieval_context(results: list[RetrievalResult]) -> str:
    """便捷函数：格式化检索结果。"""
    return get_retriever().format_context(results)
