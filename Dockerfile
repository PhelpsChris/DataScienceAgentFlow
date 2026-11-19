# Cloud Run image: serves the ADK chat UI (adk web) on $PORT.
FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agents ./agents
COPY sample_data ./sample_data
ENV OUTPUT_DIR=/tmp/runs PORT=8080 PYTHONUNBUFFERED=1 LIVE_GRAPH=0
CMD ["sh", "-c", "adk web agents --host 0.0.0.0 --port ${PORT}"]
