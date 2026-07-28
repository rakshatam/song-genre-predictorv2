# Song Genre Prediction MLOps Pipeline

This repository contains the backend and frontend components for a machine learning pipeline that predicts music genres and associated audio tags using the Audio Spectrogram Transformer (AST). The architecture is designed to decouple the static frontend from the heavy machine learning backend, allowing the model to run on dedicated hardware while serving users via a serverless edge network.

## Architecture Overview

The system utilizes a hybrid deployment architecture:

```text
    [ Client Browser ]
           │
           ▼
[ Cloudflare Pages (Frontend) ] ──(Static HTML/JS/CSS)
           │
           │ (HTTPS / Fetch API)
           ▼
    [ Ngrok Edge Network ] ──(Bypasses NAT/Firewalls)
           │
           │ (Reverse Proxy Tunnel)
           ▼
[ Local Host Environment ]
           │
           ▼
  [ Docker Compose Stack ] 
           │
           ├─ FastAPI (REST API & Inference)
           ├─ Prometheus (Metrics Scraper)
           └─ Grafana (Visual Dashboard)
```

## Core Components

### 1. PyTorch AST Model (Backend)
- Uses the `MIT/ast-finetuned-audioset-10-10-0.4593` model from Hugging Face.
- **Audio Processing**: Media files (MP4, MP3, WAV) are processed using `moviepy` and `librosa`. Audio is resampled to 16,000 Hz and divided into 10.24-second chunks (the native input size for AST).
- **Ontology Filtering**: The model outputs predictions against the 527-class AudioSet ontology. The backend implements a strict index-based blocklist to filter out non-musical environmental noise (e.g., vehicles, weather) and isolated speech, retaining only musical genres, instruments, and vocal music.

### 2. FastAPI (API Layer)
- Provides a RESTful POST endpoint (`/predict`) to handle file uploads.
- Implements HTTP Basic Authentication.
- Configured with strict CORS middleware to accept requests exclusively from the designated Cloudflare Pages frontend.

### 3. Static Frontend (Edge)
- A lightweight, static HTML/JS interface hosted on Cloudflare Pages.
- Handles user authentication and multipart/form-data file uploads.
- Communicates with the backend API via dynamic tunneling URLs.

### 4. Monitoring Stack (Local Observability)
- **Prometheus**: Automatically scrapes the `/metrics` endpoint on the FastAPI backend every 5 seconds to track CPU usage, prediction latency, and error rates.
- **Grafana**: Visualizes the Prometheus metrics on a live dashboard running locally.

### 5. Kubernetes Scaling (Infrastructure-as-Code)
- The repository includes production-ready Kubernetes manifests (`k8s/`) including a `Deployment`, `LoadBalancer Service`, and a `HorizontalPodAutoscaler` (HPA).
- *Note: We intentionally rely on `docker-compose` for the local environment because running a full local Kubernetes node (like Minikube) alongside the PyTorch model exceeds typical consumer hardware RAM limits. The `k8s/` files are provided specifically for cloud migration and for security purposes k8s/ are not available in the repos.*

## Workflow

1. **Upload**: The user uploads an audio or video file via the Cloudflare Pages interface.
2. **Pre-flight**: The browser sends an `OPTIONS` request to the backend. FastAPI's CORS middleware validates the request.
3. **Authentication**: The browser transmits the file via `POST` with HTTP Basic Auth headers.
4. **Extraction**: The FastAPI backend saves the file temporarily. If it is a video file, `moviepy` extracts the audio track.
5. **Inference**: The audio is loaded into memory, chunked into 10.24s segments, and passed through the AST feature extractor and PyTorch model.
6. **Aggregation**: The predictions for each chunk are aggregated, filtered against the AudioSet ontology blocklist, and the top 5 genres are returned to the client as JSON.
7. **Cleanup**: Temporary files are deleted from the container.
8. **Usage**: A dedicated wslconfig was made to make sure the CPU and RAM constraints are met

## Local Setup

### Prerequisites
- Docker Engine & Docker Compose
- Python 3.10+ (for local development)
- Ngrok (for external tunneling)

### Execution
The backend is completely containerized. Start the server using:
```bash
docker-compose up -d --build
```
Ensure your edge tunnel is configured to point to the designated local port (the ports and others are pointed as 'xxxx' please change accordingly to your designated port made in code , docker, Prometheus and index and docker files).
