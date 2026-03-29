# Use a lean Python 3.10 base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file explicitly
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# IMPORTANT HF HACK: The default pip `torch` installs a 2.5GB CUDA GPU version. 
# HF standard spaces ONLY use CPUs. The CPU version of PyTorch is ~300MB.
# This extracts torch/torchvision from requirements.txt so we can install a slim version separately.
RUN grep -vE "^torch(vision)?($|>=|==)" requirements.txt > reqs_cpu.txt

# Install PyTorch explicitly compiled for CPU to prevent Out-Of-Memory crashes
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
RUN pip install --no-cache-dir -r reqs_cpu.txt

# Copy your actual backend application code
COPY api/ /code/api/
COPY src/ /code/src/
COPY models_saved/ /code/models_saved/

# Hugging Face Spaces listens uniformly on Port 7860
EXPOSE 7860

# We invoke `uvicorn` explicitly, overriding your 8000 mapped in `api/main.py`
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
