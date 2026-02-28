# Triton DSL Inference Server
# Multi-stage build for optimized deployment

# Build stage
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY examples/deployment/docker_deployment/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY . /build/triton-dsl

# Install Triton DSL
WORKDIR /build/triton-dsl
RUN pip install --no-cache-dir --user -e .

# Runtime stage
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY examples/deployment/docker_deployment/app.py .
COPY models/ /app/models/
COPY backend/ /app/backend/
COPY compiler/ /app/compiler/

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Create directory for models
RUN mkdir -p /app/saved_models

# Expose port for Flask app
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Set environment variables
ENV FLASK_APP=app.py
ENV MODEL_PATH=/app/saved_models/model.pt
ENV NUM_WORKERS=4

# Run the application
CMD ["python", "app.py"]
