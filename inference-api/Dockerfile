# Use a stable, official Python base image
FROM python:3.14-slim

# Set environment variables
# PYTHONUNBUFFERED=1 ensures console logs are printed immediately
# PYTHONDONTWRITEBYTECODE=1 prevents python from writing .pyc files
# PORT=7860 is the default port for Hugging Face Spaces
# HOME=/home/user sets the home folder for the non-root user
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOST=0.0.0.0 \
    HOME=/home/user

# Install system dependencies required by OpenCV, MediaPipe, and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libgles2 \
    libegl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    sed \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user with UID 1000 (Hugging Face Spaces runs as UID 1000)
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy requirements.txt first for build caching
COPY --chown=user:user requirements.txt /app/

# Remove the custom local PyTorch wheels from requirements.txt to install standard stable versions
# from PyPI, supporting both CPU and GPU workloads automatically.
RUN sed -i '/torch==/d' requirements.txt && \
    sed -i '/torchvision==/d' requirements.txt && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY --chown=user:user . /app/

# Setup a writeable Hugging Face cache directory inside the home folder of user 1000
RUN mkdir -p /home/user/.cache/huggingface && chown -R user:user /home/user

# Switch to the non-root user
USER user

# Expose the default port (Hugging Face Spaces automatically forwards traffic to 7860)
EXPOSE 7860

# Run the FastAPI server
CMD ["python", "main.py"]
