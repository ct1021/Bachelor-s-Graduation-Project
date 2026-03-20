FROM python:3.10-slim

WORKDIR /app

# 仅安装 Dashboard 所需的 Python 依赖 (无需 GL/MuJoCo)
COPY requirements-dashboard.txt .
RUN pip install --no-cache-dir -r requirements-dashboard.txt

# 只复制 Dashboard 和必要的静态资源
COPY g1_dashboard/ ./g1_dashboard/
COPY envs/ ./envs/

EXPOSE 8000

CMD ["uvicorn", "g1_dashboard.server:app", "--host", "0.0.0.0", "--port", "8000"]
