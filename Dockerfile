FROM python:3.10-slim

# Install system dependencies required for audio processing (ffmpeg)
# and clean up apt cache to keep image small
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# We manipulate the requirements file here to ensure we install the CPU-only version of PyTorch.
# This prevents Docker from downloading the 2GB+ CUDA binaries which we don't need!
RUN sed -i 's/cu121/cpu/g' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application files
# Note: Thanks to .dockerignore, this won't copy your gigabytes of MP4 test videos,
# but it WILL copy the `model_cache` folder so the AI model boots instantly!
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run Uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
