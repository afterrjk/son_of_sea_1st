"""知识库检索模块 —— 本地 RAG 检索，双后端实现。

提供：
- init_knowledge_base()：从 JSON 文件加载知识条目并向量化存储
- search()：根据用户查询检索最相关的知识条目
- get_retrieval_context()：将检索结果格式化为可直接注入系统提示词的文本

后端选择：
- ChromaDB（Linux / Docker 环境默认，性能与扩展性更好）
- 纯 numpy + JSON（Windows 环境自动回退，避免 ChromaDB hnswlib 在
  Windows 上已知的内存访问崩溃问题）

可通过环境变量 RAG_BACKEND=chromadb|numpy 强制指定后端。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypedDict

from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
NUMPY_STORE_DIR = BASE_DIR / "data" / "numpy_store"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# 优先使用随项目打包的本地模型目录，避免在离线/受限网络环境访问 HuggingFace
LOCAL_MODEL_DIR = BASE_DIR / "models" / "all-MiniLM-L6-v2"
COLLECTION_NAME = "zhuan_safety_knowledge"

# Windows 上 ChromaDB 的 hnswlib 存在已知的 0xC0000005 崩溃缺陷，
# 默认回退到纯 numpy 实现；Linux / Docker 继续使用 ChromaDB。
_ENV_BACKEND = (os.getenv("RAG_BACKEND") or "").strip().lower()
if _ENV_BACKEND == "chromadb":
    _USE_CHROMADB = True
elif _ENV_BACKEND == "numpy":
    _USE_CHROMADB = False
else:
    _USE_CHROMADB = os.name != "nt"

if _USE_CHROMADB:
    import chromadb
    from chromadb.config import Settings


class RetrievalResult(TypedDict):
    """单条检索结果。"""

    id: str
    category: str
    title: str
    content: str
    keywords: list[str]
    score: float


def _build_embedding_text(entry: dict) -> str:
    """构造用于向量化的文本：标题 + 关键词 + 正文（截断以控制嵌入质量）。"""
    kw_str = "，".join(entry.get("keywords", []))
    content = entry.get("content", "")
    return (
        f"【{entry['category']}】{entry['title']}\n"
        f"关键词：{kw_str}\n"
        f"{content[:1500]}"
    )


class NumpyKnowledgeStore:
    """基于 numpy + JSON 文件的轻量向量存储，替代 ChromaDB。

    数据结构：
    - numpy_store/index.json：全部知识条目的元数据（id/category/title/content/keywords）
    - numpy_store/vectors.npy：与 index.json 顺序一致的嵌入向量矩阵 (N, dim)
    """

    def __init__(self, store_dir: Path) -> None:
        self.store_dir = store_dir
        self.index_path = store_dir / "index.json"
        self.vectors_path = store_dir / "vectors.npy"
        self._entries: list[dict] = []
        self._vectors = None
        self._load()

    def _load(self) -> None:
        if self.index_path.exists() and self.vectors_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    self._entries = json.load(f)
                import numpy as np

                self._vectors = np.load(self.vectors_path)
            except Exception:
                self._entries = []
                self._vectors = None

    def count(self) -> int:
        return len(self._entries)

    def save(self, entries: list[dict], vectors) -> None:
        import numpy as np

        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._entries = entries
        self._vectors = np.asarray(vectors, dtype=np.float32)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        np.save(self.vectors_path, self._vectors)

    def query(
        self, query_vector, top_k: int, category_filter: str | None = None
    ) -> list[RetrievalResult]:
        """余弦相似度检索，返回按相似度降序的结果。"""
        import numpy as np

        if self.count() == 0 or self._vectors is None:
            return []

        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        # 归一化后点积 = 余弦相似度
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        mat_norm = self._vectors / (
            np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-9
        )
        scores = mat_norm @ q_norm.T
        scores = scores.ravel()

        idxs = np.argsort(scores)[::-1]
        results: list[RetrievalResult] = []
        for i in idxs:
            if len(results) >= top_k:
                break
            entry = self._entries[i]
            if category_filter and entry.get("category") != category_filter:
                continue
            results.append(
                {
                    "id": entry["id"],
                    "category": entry.get("category", ""),
                    "title": entry.get("title", ""),
                    "content": entry.get("content", ""),
                    "keywords": entry.get("keywords", []),
                    "score": round(float(scores[i]), 4),
                }
            )
        return results


class KnowledgeRetriever:
    """知识库检索器，封装向量化、存储和查询功能（支持 ChromaDB / numpy 双后端）。"""

    def __init__(self) -> None:
        self._embedding_model: SentenceTransformer | None = None
        self._chroma_client = None
        self._collection = None
        self._numpy_store: NumpyKnowledgeStore | None = None
        self.backend = "chromadb" if _USE_CHROMADB else "numpy"

    @property
    def embedding_model(self) -> SentenceTransformer:
        """延迟加载向量模型，优先加载本地打包模型，避免联网下载。"""
        if self._embedding_model is None:
            if LOCAL_MODEL_DIR.exists():
                self._embedding_model = SentenceTransformer(str(LOCAL_MODEL_DIR))
            else:
                self._embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self._embedding_model

    @property
    def chroma_client(self):
        if self._chroma_client is None:
            os.makedirs(str(CHROMA_DIR), exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
        return self._chroma_client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "筑安建筑安全知识库"},
            )
        return self._collection

    @property
    def numpy_store(self) -> NumpyKnowledgeStore:
        if self._numpy_store is None:
            self._numpy_store = NumpyKnowledgeStore(NUMPY_STORE_DIR)
        return self._numpy_store

    def is_initialized(self) -> bool:
        """检查知识库是否已经完成向量化入库。"""
        if self.backend == "chromadb":
            try:
                return self.collection.count() > 0
            except Exception:
                return False
        return self.numpy_store.count() > 0

    def init_from_json(self, json_path: str | None = None) -> int:
        """从 JSON 文件加载知识条目，逐条向量化并存入后端存储。

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

        texts = [_build_embedding_text(entry) for entry in entries]

        # 生成向量
        embeddings = self.embedding_model.encode(
            texts, show_progress_bar=True, convert_to_numpy=True
        )

        if self.backend == "chromadb":
            ids = [entry["id"] for entry in entries]
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
                    embeddings=embeddings[batch_slice].tolist(),
                    metadatas=metadatas[batch_slice],
                    documents=documents[batch_slice],
                )
            return total

        # numpy 后端：直接保存条目与向量
        store_entries = [
            {
                "id": entry["id"],
                "category": entry["category"],
                "title": entry["title"],
                "content": entry["content"],
                "keywords": entry.get("keywords", []),
            }
            for entry in entries
        ]
        self.numpy_store.save(store_entries, embeddings)
        return len(entries)

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
        )

        if self.backend == "chromadb":
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
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
                similarity = max(0.0, min(1.0, 1.0 - distance))

                keywords_raw = metadata.get("keywords", "")
                keywords = (
                    [kw.strip() for kw in keywords_raw.split(",") if kw.strip()]
                    if keywords_raw
                    else []
                )

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

        return self.numpy_store.query(query_embedding, top_k=top_k)

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

    if retriever.backend == "numpy":
        query_embedding = retriever.embedding_model.encode(
            [query], show_progress_bar=False, convert_to_numpy=True
        )
        return retriever.numpy_store.query(query_embedding, top_k=top_k, category_filter=category_filter)

    # ChromaDB 后端
    query_embedding = retriever.embedding_model.encode(
        [query], show_progress_bar=False, convert_to_numpy=True
    ).tolist()

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
