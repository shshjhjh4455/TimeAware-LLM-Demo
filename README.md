# TimeAware LLM Role-Play Demo

Time-aware character role-playing system that enables LLMs to answer questions as Harry Potter characters while respecting narrative timeline constraints.

The system uses a **Hypothesis-Verification RAG pipeline** -- a RoBERTa-based Scene Navigator predicts candidate chapters, a GPT-powered Verifier selects the correct scene, and a Memory Retriever provides temporally-filtered character knowledge for grounded, in-character responses.

## Architecture

```
User Question
     |
     v
[Scene Navigator]  RoBERTa-Large (GPU) - Top-k chapter prediction
     |
     v
[Scene Verifier]   GPT-5-mini - Selects best scene from candidates
     |                          (runs in parallel with Memory Retriever)
     v
[Memory Retriever] Vector similarity search - Temporal knowledge filtering
     |
     v
[Response Generator] GPT-5-mini - Premise-aware, dual-timeline generation
     |
     v
[SSE Stream] -----> React Frontend (real-time token streaming)
```

## Verified Test Results

All components have been tested end-to-end with `docker compose up --build`.

| Component | Status | Details |
|-----------|--------|---------|
| Backend image build | OK | PyTorch 2.1 + CUDA 12.1 base, ~3 min (cached) |
| Frontend image build | OK | Node 20 build + Nginx serve, ~25 sec |
| Scene Navigator (GPU) | OK | RoBERTa-Large, 185 chapter classes, CUDA inference |
| Memory Retriever | OK | 7,034 character memories loaded |
| Vector Knowledge Retriever | OK | 7,016 knowledge embeddings loaded |
| Search Index | OK | 988 scene embeddings loaded |
| Pipeline initialization | OK | ~40 sec total (including model load to GPU) |
| `GET /api/health` | OK | `{"pipeline_ready": true, "status": "ok"}` |
| `GET /api/characters` | OK | 3 characters (Harry, Hermione, Ron) x 25 time periods |
| `POST /api/chat` | OK | Full inference pipeline verified |

### Example API Response

```
POST /api/chat
{
  "character": "Harry Potter",
  "character_period": "1st-year / on halloween",
  "question": "What do you think about Hermione?"
}

-> Navigator: Top-3 hypotheses (Book6-ch8, Book7-ch17, Book5-ch4)
-> Verifier: Selected best matching scene
-> Memory: Temporally filtered knowledge (pre-Halloween 1st year)
-> Response: In-character answer respecting timeline constraints
```

## Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA support + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- OpenAI API key (GPT-5-mini)
- Git LFS

## Quick Start

### 1. Install NVIDIA Container Toolkit (if not installed)

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU is accessible from Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 2. Clone with Git LFS

```bash
git lfs install
git clone https://github.com/shshjhjh4455/TimeAware-LLM-Demo.git
cd TimeAware-LLM-Demo
git lfs pull
```

Verify LFS files are downloaded (not pointer files):
```bash
git lfs ls-files
# Should show LFS-tracked files:
#   knowledge_embeddings_array.npy (165MB)
#   knowledge_embeddings.pkl (168MB)
#   memory_vectors.npy (83MB)
#   search_index_embeddings.npy (12MB)
#   roberta_200q_best_20251010_064944.pt (1.4GB)
```

### 3. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 4. Build and run

```bash
docker compose up --build
```

First build takes ~10 minutes (PyTorch base image + dependencies download).
Pipeline initialization takes ~40 seconds (RoBERTa model loading to GPU + database loading).

Watch for this log line to confirm the backend is ready:
```
webapp.backend.pipeline_manager: Pipeline initialized in XX.Xs
webapp.backend.app: Pipeline ready. Starting Flask server.
```

### 5. Access the app

Open **http://localhost** in your browser.

1. Select a character (Harry Potter, Hermione Granger, or Ron Weasley)
2. Choose a time period (e.g., "1st-year / on Halloween")
3. Ask a question about events in the Harry Potter story
4. Click the debug panel to see Navigator, Verifier, and Memory Retriever outputs

## Docker Configuration

### docker-compose.yml

Both services use `network_mode: host` for reliable GPU networking:

| Service | Base Image | Port | GPU |
|---------|-----------|------|-----|
| backend | `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` | 5000 | Yes (`runtime: nvidia`) |
| frontend | `node:20-alpine` (build) + `nginx:alpine` (serve) | 80 | No |

### Volumes (backend)

The backend mounts these directories from the host to avoid copying large files into the image:

| Host Path | Container Path | Contents |
|-----------|---------------|----------|
| `./knowledge_enriched_rag/` | `/app/knowledge_enriched_rag/` | RAG pipeline code + databases + model |
| `./timechara/` | `/app/timechara/` | Character period mapping utilities |
| `./.env` | `/app/.env` | API key |

### Useful commands

```bash
# Run in background
docker compose up --build -d

# View backend logs
docker compose logs -f backend

# Stop
docker compose down

# Rebuild backend only
docker compose up --build backend
```

## Running without Docker

### Backend

```bash
# Requires: Python 3.10+, CUDA-capable GPU, PyTorch
pip install -r requirements.txt
pip install -r webapp/backend/requirements.txt

cp .env.example .env  # Add your API key

PYTHONPATH=. python -m webapp.backend.app
# Backend runs at http://localhost:5000
```

### Frontend

```bash
cd webapp/frontend
npm install
npm run dev
# Frontend runs at http://localhost:5173 (Vite dev server)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Pipeline status check |
| GET | `/api/characters` | Available characters and time periods |
| POST | `/api/chat` | Synchronous inference |
| POST | `/api/chat/stream` | SSE streaming inference |

### Example: POST /api/chat/stream

```bash
curl -N -X POST http://localhost:5000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "character": "Harry Potter",
    "character_period": "1st-year / on halloween",
    "question": "What happened with the troll?"
  }'
```

## Project Structure

```
TimeAware-LLM-Demo/
|-- docker-compose.yml
|-- requirements.txt
|-- .env.example
|
|-- webapp/
|   |-- backend/
|   |   |-- app.py                  # Flask API (REST + SSE streaming)
|   |   |-- config.py               # Pipeline & character configuration
|   |   |-- pipeline_manager.py     # Singleton pipeline orchestrator
|   |   +-- Dockerfile              # pytorch/pytorch CUDA base image
|   +-- frontend/
|       |-- src/components/         # React components (Chat, DebugPanel, etc.)
|       +-- Dockerfile              # Multi-stage: node build + nginx serve
|
|-- knowledge_enriched_rag/
|   |-- inference_pipeline.py       # Main RAG pipeline (dual-timeline prompts)
|   |-- scene_classifier.py         # RoBERTa Scene Navigator (GPU)
|   |-- verifier.py                 # GPT Scene Verifier
|   |-- memory_retriever.py         # Vector similarity memory search
|   |-- processors/                 # Knowledge filter & vector retriever
|   |-- databases/                  # Pre-built HP knowledge databases
|   |   |-- character_knowledge.json          # 7,034 character knowledge items
|   |   |-- scenes_vector.json                # 988 scene metadata
|   |   |-- memory_vectors.npy                # Memory embeddings (83MB, LFS)
|   |   |-- memory_metadata.json              # Memory metadata (3.3MB)
|   |   |-- search_index_embeddings.npy       # Scene search embeddings (12MB, LFS)
|   |   |-- knowledge_embeddings_array.npy    # Knowledge embeddings (165MB, LFS)
|   |   +-- knowledge_embeddings.pkl          # Knowledge embeddings pickle (168MB, LFS)
|   +-- models/
|       +-- roberta_200q_best_*.pt  # Scene Navigator weights (1.4GB, LFS)
|
+-- timechara/
    +-- utils.py                    # Character period mapping
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4GB (RoBERTa-Large) | 8GB+ |
| RAM | 8GB | 16GB+ |
| Disk | 4GB (with LFS files) | 4GB |

## License

This project is for research and educational purposes.
