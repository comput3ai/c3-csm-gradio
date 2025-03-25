# Base image with CUDA runtime
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Disable Triton compilation
ENV NO_TORCH_COMPILE=1

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    ffmpeg \
    python3-pip \
    python3-dev \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -m -s /bin/bash appuser

# Create app directory with appropriate permissions
RUN mkdir -p /app && chown appuser:appuser /app

# Set working directory
WORKDIR /app

# Switch to non-root user
USER appuser

# Set up a Python virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy the CSM directory
COPY --chown=appuser:appuser csm /app/csm

# Install CSM requirements
RUN pip install --no-cache-dir -r /app/csm/requirements.txt

# Copy requirements file
COPY --chown=appuser:appuser requirements.txt /app/

# Install Gradio and other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser app.py /app/csm/

# Expose the port for the Gradio web interface
EXPOSE 7860

# Change working directory to csm
WORKDIR /app/csm

# Command to run when the container starts
CMD ["python", "app.py"]