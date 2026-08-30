#!/usr/bin/env bash
set -e

# Cloud Studio 启动脚本：
# 1. 若本地模型不存在，从 ModelScope 国内镜像下载完整向量模型
# 2. 以离线优先方式启动 uvicorn
MODEL_DIR=models/all-MiniLM-L6-v2
BASE_URL="https://modelscope.cn/models/AI-ModelScope/all-MiniLM-L6-v2/resolve/master"

if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
  echo "[run.sh] downloading embedding model from ModelScope..."
  mkdir -p "$MODEL_DIR/1_Pooling"
  for f in config.json config_sentence_transformers.json model.safetensors modules.json README.md sentence_bert_config.json special_tokens_map.json tokenizer.json tokenizer_config.json vocab.txt; do
    curl -sSL --retry 3 --retry-delay 2 -o "$MODEL_DIR/$f" "$BASE_URL/$f" || { echo "download failed: $f"; exit 1; }
  done
  curl -sSL --retry 3 --retry-delay 2 -o "$MODEL_DIR/1_Pooling/config.json" "$BASE_URL/1_Pooling/config.json" || { echo "download failed: 1_Pooling/config.json"; exit 1; }
  echo "[run.sh] model downloaded."
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
python -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-8001}"
