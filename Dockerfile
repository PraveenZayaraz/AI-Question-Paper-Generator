# Stage 1: build the React frontend into static files.
FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: the Python API, which also serves the built frontend.
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    QPG_DATA_DIR=/data \
    QPG_STATIC_DIR=/app/frontend/dist
COPY requirements.txt requirements-core.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY qpg/ ./qpg/
COPY examples/ ./examples/
COPY --from=frontend /app/frontend/dist ./frontend/dist
VOLUME ["/data"]
EXPOSE 8000
CMD ["uvicorn", "qpg.api:app", "--host", "0.0.0.0", "--port", "8000"]
