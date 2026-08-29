#!/usr/bin/env bash
set -e

# Cloud Studio / Linux 云服务器的一键启动脚本。
python -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-8001}"

