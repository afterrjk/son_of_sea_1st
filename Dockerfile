# 筑安云端运行镜像：适用于 Cloud Studio、腾讯云轻量服务器等 Linux 环境。
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先安装依赖，便于后续修改业务代码时复用 Docker 缓存。
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY data ./data
COPY static ./static
COPY main.py README.md .env.example ./

EXPOSE 8001

# 监听 0.0.0.0 才能被 Cloud Studio 或公网反向代理访问。
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

