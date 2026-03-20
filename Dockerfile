FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖 (MuJoCo 需要 libgl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements-dashboard.txt .
RUN pip install --no-cache-dir -r requirements-dashboard.txt

# 复制项目
COPY . .

EXPOSE 8000

CMD ["uvicorn", "g1_dashboard.server:app", "--host", "0.0.0.0", "--port", "8000"]
