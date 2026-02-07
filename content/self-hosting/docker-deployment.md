# Docker Deployment for Self-Hosted LLMs

Containerizing your LLM deployment provides reproducibility, isolation, and easier scaling. This guide covers Docker best practices for LLM serving.

## Why Docker for LLMs?

```yaml
docker_benefits:
  consistency:
    - "Same environment across dev/staging/prod"
    - "Eliminates 'works on my machine' issues"
    - "Version-controlled infrastructure"
    
  isolation:
    - "Separate model versions easily"
    - "Clean dependency management"
    - "Resource limits and control"
    
  scaling:
    - "Easy horizontal scaling"
    - "Kubernetes-ready deployments"
    - "Load balancing friendly"
    
  management:
    - "Simple updates and rollbacks"
    - "Health checks built-in"
    - "Logging and monitoring integration"
```

## Ollama Docker Deployment

### Basic Setup

```dockerfile
# docker-compose.yml for Ollama
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    # CPU only
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/version"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  ollama_data:
```

### With NVIDIA GPU

```yaml
# docker-compose.yml with GPU support
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-gpu
    volumes:
      - ollama_data:/root/.ollama
      - ./models:/models  # Optional: pre-downloaded models
    ports:
      - "11434:11434"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    restart: unless-stopped

volumes:
  ollama_data:
```

### Pre-loading Models

```dockerfile
# Dockerfile with pre-loaded models
FROM ollama/ollama:latest

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
```

```bash
#!/bin/bash
# entrypoint.sh

# Start Ollama in background
ollama serve &

# Wait for Ollama to be ready
sleep 5

# Pull required models
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# Keep container running
wait
```

## vLLM Docker Deployment

### Basic vLLM Setup

```yaml
# docker-compose.yml for vLLM
version: '3.8'

services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm-server
    volumes:
      - huggingface_cache:/root/.cache/huggingface
    ports:
      - "8000:8000"
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    command: >
      --model meta-llama/Llama-3.2-3B-Instruct
      --host 0.0.0.0
      --port 8000
      --max-model-len 4096
      --gpu-memory-utilization 0.9
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

volumes:
  huggingface_cache:
```

### Multi-GPU vLLM

```yaml
# docker-compose.yml for multi-GPU
version: '3.8'

services:
  vllm-70b:
    image: vllm/vllm-openai:latest
    container_name: vllm-70b
    volumes:
      - huggingface_cache:/root/.cache/huggingface
    ports:
      - "8000:8000"
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
      - CUDA_VISIBLE_DEVICES=0,1,2,3
    command: >
      --model meta-llama/Llama-3.1-70B-Instruct
      --host 0.0.0.0
      --port 8000
      --tensor-parallel-size 4
      --max-model-len 8192
      --gpu-memory-utilization 0.95
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 4
              capabilities: [gpu]
    ipc: host  # Required for multi-GPU
    restart: unless-stopped

volumes:
  huggingface_cache:
```

## Production-Ready Setup

### Complete Stack

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  # LLM Server
  llm-server:
    image: vllm/vllm-openai:latest
    container_name: llm-server
    volumes:
      - huggingface_cache:/root/.cache/huggingface
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    command: >
      --model ${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}
      --host 0.0.0.0
      --port 8000
      --max-model-len ${MAX_MODEL_LEN:-4096}
      --gpu-memory-utilization ${GPU_UTIL:-0.9}
      --api-key ${API_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    networks:
      - llm-network

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: nginx-proxy
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    ports:
      - "443:443"
      - "80:80"
    depends_on:
      - llm-server
    restart: unless-stopped
    networks:
      - llm-network

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - llm-network

  # Grafana Dashboard
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    networks:
      - llm-network

volumes:
  huggingface_cache:
  prometheus_data:
  grafana_data:

networks:
  llm-network:
    driver: bridge
```

### Nginx Configuration

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream llm_backend {
        server llm-server:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=llm_limit:10m rate=10r/s;

    server {
        listen 80;
        server_name _;
        return 301 https://$host$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name _;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        location /v1/ {
            limit_req zone=llm_limit burst=20 nodelay;
            
            proxy_pass http://llm_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            
            # Streaming support
            proxy_buffering off;
            proxy_cache off;
            
            # Long timeout for LLM responses
            proxy_read_timeout 300s;
            proxy_send_timeout 300s;
        }

        location /health {
            proxy_pass http://llm_backend/health;
        }

        location /metrics {
            proxy_pass http://llm_backend/metrics;
            allow 10.0.0.0/8;
            deny all;
        }
    }
}
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['llm-server:8000']
    metrics_path: /metrics
```

## Custom Dockerfile Examples

### llama.cpp Server

```dockerfile
# Dockerfile.llamacpp
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as builder

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /build/llama.cpp

RUN cmake -B build -DGGML_CUDA=ON
RUN cmake --build build --config Release -j$(nproc)

FROM nvidia/cuda:12.1-runtime-ubuntu22.04

COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/
COPY --from=builder /build/llama.cpp/build/bin/llama-cli /usr/local/bin/

RUN mkdir /models

EXPOSE 8080

ENTRYPOINT ["llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
```

### Multi-Model Server

```dockerfile
# Dockerfile.multimodel
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    openai \
    httpx

COPY server.py /app/

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```python
# server.py - Multi-model router
from fastapi import FastAPI, HTTPException
from openai import OpenAI
import os

app = FastAPI()

# Model routing configuration
MODELS = {
    "small": {"endpoint": "http://ollama:11434/v1", "model": "llama3.2:3b"},
    "medium": {"endpoint": "http://vllm-8b:8000/v1", "model": "llama-3.1-8b"},
    "large": {"endpoint": "http://vllm-70b:8000/v1", "model": "llama-3.1-70b"},
}

def get_client(model_size: str) -> tuple[OpenAI, str]:
    config = MODELS.get(model_size, MODELS["small"])
    client = OpenAI(base_url=config["endpoint"], api_key="local")
    return client, config["model"]

@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    # Route based on complexity or explicit model choice
    model_size = request.pop("model_size", "small")
    client, model = get_client(model_size)
    
    request["model"] = model
    response = client.chat.completions.create(**request)
    return response.model_dump()
```

## Resource Management

### GPU Memory Limits

```yaml
# Limit GPU memory per container
services:
  llm-1:
    image: vllm/vllm-openai:latest
    environment:
      - CUDA_VISIBLE_DEVICES=0
    command: --gpu-memory-utilization 0.5  # Use only 50%
    
  llm-2:
    image: vllm/vllm-openai:latest
    environment:
      - CUDA_VISIBLE_DEVICES=0
    command: --gpu-memory-utilization 0.5  # Use other 50%
```

### CPU and Memory Limits

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 32G
        reservations:
          cpus: '4'
          memory: 16G
```

## Health Checks and Monitoring

```python
# healthcheck.py
import requests
import sys

def check_health(url: str, timeout: int = 10) -> bool:
    try:
        response = requests.get(f"{url}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False

def check_model_loaded(url: str, model: str) -> bool:
    try:
        response = requests.get(f"{url}/v1/models", timeout=10)
        models = response.json().get("data", [])
        return any(m["id"] == model for m in models)
    except:
        return False

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    if check_health(url):
        sys.exit(0)
    sys.exit(1)
```

## Deployment Script

```bash
#!/bin/bash
# deploy.sh

set -e

# Configuration
export MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.2-3B-Instruct"}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
export GPU_UTIL=${GPU_UTIL:-0.9}
export API_KEY=${API_KEY:-$(openssl rand -hex 32)}

echo "Deploying LLM server with model: $MODEL_NAME"

# Pull latest images
docker compose -f docker-compose.production.yml pull

# Start services
docker compose -f docker-compose.production.yml up -d

# Wait for health check
echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -sf http://localhost:8000/health > /dev/null; then
        echo "Server is ready!"
        exit 0
    fi
    echo "Waiting... ($i/60)"
    sleep 5
done

echo "Server failed to start"
docker compose -f docker-compose.production.yml logs
exit 1
```

## Summary

```
┌─────────────────────────────────────────────────────────────┐
│              DOCKER LLM DEPLOYMENT CHECKLIST                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  □ Choose deployment type:                                  │
│    - Ollama: Simple, good for dev/small prod               │
│    - vLLM: High throughput production                      │
│    - llama.cpp: Maximum control                            │
│                                                             │
│  □ Configure GPU support:                                   │
│    - Install NVIDIA Container Toolkit                       │
│    - Add GPU reservations to compose file                  │
│                                                             │
│  □ Production setup:                                        │
│    - Reverse proxy (Nginx)                                 │
│    - SSL/TLS termination                                   │
│    - Rate limiting                                         │
│    - Health checks                                         │
│    - Monitoring (Prometheus/Grafana)                       │
│                                                             │
│  □ Resource management:                                     │
│    - Set GPU memory utilization                            │
│    - Configure CPU/memory limits                           │
│    - Plan for scaling                                      │
│                                                             │
│  □ Security:                                                │
│    - API key authentication                                │
│    - Network isolation                                     │
│    - Secrets management                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Next Steps

1. **Integration Patterns** - Connect Docker deployment to apps
2. **Kubernetes** - Scale with K8s
3. **Monitoring** - Set up observability
