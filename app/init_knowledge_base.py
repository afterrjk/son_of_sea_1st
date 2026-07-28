r"""知识库初始化脚本。

在首次部署或知识库 JSON 更新后运行一次，将知识条目向量化并存入 ChromaDB。

使用方法：
    cd E:\AI agent\son_of_sea
    .\.venv\Scripts\Activate.ps1
    python -m app.init_knowledge_base

首次运行会自动下载 sentence-transformers/all-MiniLM-L6-v2 模型（约 90MB）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.retrieval import KnowledgeRetriever


def main() -> None:
    json_path = BASE_DIR / "data" / "knowledge_base.json"
    if not json_path.exists():
        print(f"[ERROR] 知识库文件不存在：{json_path}")
        print("   请确认 knowledge_base.json 已放置在 data/ 目录下。")
        sys.exit(1)

    print("=" * 60)
    print("  筑安知识库初始化")
    print("=" * 60)
    print(f"\n[知识库文件] {json_path}")
    print(f"[嵌入模型] sentence-transformers/all-MiniLM-L6-v2")
    print(f"[向量数据库] ChromaDB (持久化)")
    print()

    start = time.perf_counter()

    try:
        retriever = KnowledgeRetriever()
        count = retriever.init_from_json(str(json_path))
        elapsed = time.perf_counter() - start

        print(f"\n[OK] 初始化完成！")
        print(f"   入库条目：{count} 条")
        print(f"   耗时：{elapsed:.1f} 秒")
        print(f"   存储位置：{BASE_DIR / 'data' / 'chroma_db'}")

        # 快速验证：执行一次测试检索
        print("\n[测试检索] 查询: 高处作业安全带怎么用")
        results = retriever.search("高处作业安全带怎么用", top_k=3)
        if results:
            print("  命中条目：")
            for r in results:
                print(f"   - [{r['score']:.3f}] {r['category']} > {r['title']}")
        else:
            print("  [WARNING] 测试检索无结果，请检查数据。")
    except Exception as exc:
        print(f"\n[ERROR] 初始化失败：{exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
