# TTS V2 API Server

![Docker Build](https://github.com/bmv234/TTS-V2-API-Server/actions/workflows/docker-build.yml/badge.svg)

A FastAPI-based Text-to-Speech API service using the Kokoro-82M model. Supports multiple voices in American and British English accents, with OpenAI API compatibility.

## Features

- Multiple voices (American and British English)
- Easy-to-use REST API
- OpenAI API compatibility
- Web interface
- Automatic model file management
- Proper WAV file generation
- Detailed API documentation

## Setup Options

### Docker (Recommended)

1. Using docker-compose:
```bash
# Build and start the service
./docker-build.sh

# Or manually:
docker-compose up --build
```

2. Using Docker directly:
```bash
# Build the image
docker build -t kokoro-tts .

# Run the container
docker run -p 8000:8000 \
  -v $(pwd)/models.py:/app/models.py \
  -v $(pwd)/kokoro.py:/app/kokoro.py \
  -v $(pwd)/istftnet.py:/app/istftnet.py \
  -v $(pwd)/plbert.py:/app/plbert.py \
  -v $(pwd)/config.json:/app/config.json \
  -v $(pwd)/kokoro-v0_19.pth:/app/kokoro-v0_19.pth \
  -v $(pwd)/voices:/app/voices \
  kokoro-tts
```

### Manual Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
# API only
python main.py

# With web interface
python main-ui.py
```

The server will automatically download required model files on first run.

## Web Interface

Access the web interface at http://localhost:8000 to:
- Convert text to speech
- Choose from available voices
- Download generated audio
- View API examples

## API Endpoints

### Standard API

#### GET /
Returns API usage information and status.

#### GET /voices
Returns list of available voices with descriptions.

#### POST /tts
Converts text to speech.

Parameters:
- `text`: Text to convert to speech
- `voice` (optional): Voice ID to use (default: "af")

Example:
```bash
# Generate speech with default voice
curl -X POST "http://localhost:8000/tts?text=Hello%20world." --output output.wav

# Generate speech with specific voice
curl -X POST "http://localhost:8000/tts?text=Hello%20world.&voice=am_adam" --output output.wav
```

### OpenAI-Compatible API

#### GET /v1/models
Lists available TTS models.

#### POST /v1/audio/speech
OpenAI-compatible endpoint for speech generation.

Example using OpenAI Python client:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key not required
)

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",  # See voice mapping below
    input="Hello world!"
)

response.stream_to_file("output.wav")
```

## Available Voices

### Standard Voices
- af: Default voice (50-50 mix of Bella & Sarah)
- af_bella: American Female - Bella
- af_sarah: American Female - Sarah
- am_adam: American Male - Adam
- am_michael: American Male - Michael
- bf_emma: British Female - Emma
- bf_isabella: British Female - Isabella
- bm_george: British Male - George
- bm_lewis: British Male - Lewis
- af_nicole: American Female - Nicole (ASMR voice)

### OpenAI Voice Mapping
- alloy → am_adam (Neutral male)
- echo → af_nicole (Soft female)
- fable → bf_emma (British female)
- onyx → bm_george (Deep male)
- nova → af_bella (Energetic female)
- shimmer → af_sarah (Clear female)

## Requirements

See `requirements.txt` for full list of dependencies.

## CI/CD

This project uses GitHub Actions for continuous integration and deployment:

### GitHub Setup

1. Branch Configuration:
   - Default branch: `main`
   - Protected branch rules recommended
   - Workflow runs on `main` branch and tags

2. Repository Settings:
   - Actions permissions: Read and write
   - Packages enabled for Docker images
   - Workflow permissions enabled

### Docker Build Workflow

The workflow automatically builds and publishes Docker images:
- Triggers on:
  * Pushes to `main` branch
  * Version tags (`v*`)
  * Pull requests
- Publishes to GitHub Container Registry (ghcr.io)
- Uses build caching for faster builds
- Includes version tagging and metadata

### Using Pre-built Images

```bash
# Pull the latest image
docker pull ghcr.io/bmv234/tts-v2-api-server:latest

# Run with GPU support
docker run --gpus all -p 8000:8000 \
  -v ./models.py:/app/models.py \
  -v ./kokoro.py:/app/kokoro.py \
  -v ./istftnet.py:/app/istftnet.py \
  -v ./plbert.py:/app/plbert.py \
  -v ./config.json:/app/config.json \
  -v ./kokoro-v0_19.pth:/app/kokoro-v0_19.pth \
  -v ./voices:/app/voices \
  ghcr.io/bmv234/tts-v2-api-server:latest
```

### Version Tags

The workflow automatically handles version tagging:
- Latest: Always points to latest main branch build
- Version tags: Created when pushing tags (e.g., v1.0.0)
- SHA tags: Include git commit hash for traceability

## Docker Configuration

The Docker setup includes:
- Multi-stage build for smaller image size
- Non-root user for security
- Health checks for container monitoring
- Volume mounts for model persistence
- Resource limits (configurable in docker-compose.yml)

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: Set to empty for CPU mode, remove for GPU support
- Memory limits: 4GB max, 2GB min (adjustable in docker-compose.yml)
