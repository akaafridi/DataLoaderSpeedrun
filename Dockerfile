FROM python:3.10-slim

# Install system dependencies including perf tools
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    linux-perf \
    linux-tools-generic \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# Install package in development mode and build C++ extension
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH="/app:$PYTHONPATH"

# Default command runs benchmark with multi-seed using the CLI tool
CMD ["grok3-io-opt", "benchmark", "--multi-seed"]