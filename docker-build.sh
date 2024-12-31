#!/bin/bash

# Stop on error
set -e

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed"
    exit 1
fi

# Check if nvidia-docker is installed
if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo "Warning: nvidia-docker runtime not found. GPU support may not be available."
    echo "Install nvidia-docker2 for GPU support."
fi

# Check for required files
required_files=(
    "main.py"
    "main-ui.py"
    "templates/index.html"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required file '$file' not found"
        exit 1
    fi
done

# Create required directories
mkdir -p voices

echo "Building Kokoro TTS Docker image..."
echo "Using CUDA 12.6.3 with GPU support"
echo "Model files will be downloaded on first run"

# Build the Docker image with progress output
docker build \
  --network=host \
  --progress=plain \
  --tag kokoro-tts:latest \
  --tag kokoro-tts:cuda12.6.3 \
  .

echo
echo "Build complete!"
echo
echo "You can run the server with:"
echo "docker run --gpus all -p 8000:8000 \\"
echo "  -v \$(pwd)/models.py:/app/models.py \\"
echo "  -v \$(pwd)/kokoro.py:/app/kokoro.py \\"
echo "  -v \$(pwd)/istftnet.py:/app/istftnet.py \\"
echo "  -v \$(pwd)/plbert.py:/app/plbert.py \\"
echo "  -v \$(pwd)/config.json:/app/config.json \\"
echo "  -v \$(pwd)/kokoro-v0_19.pth:/app/kokoro-v0_19.pth \\"
echo "  -v \$(pwd)/voices:/app/voices \\"
echo "  kokoro-tts:latest"
echo
echo "Or using docker-compose:"
echo "docker-compose up"
echo
echo "To verify GPU support:"
echo "docker run --gpus all --rm kokoro-tts:latest python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'"
echo
echo "To check server health:"
echo "curl -k https://localhost:8000/"
