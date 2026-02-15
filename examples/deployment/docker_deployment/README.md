# Docker Deployment Guide for Triton DSL

This directory contains everything needed to deploy Triton DSL models using Docker containers.

## Contents

- `Dockerfile` - Multi-stage Docker build configuration
- `app.py` - Flask-based inference API server
- `requirements.txt` - Python dependencies for deployment
- `docker-compose.yml` - Docker Compose configuration (optional)

## Quick Start

### 1. Build the Docker Image

```bash
# From the Triton project root directory
cd examples/deployment/docker_deployment

# Build the image
docker build -t triton-inference:latest -f Dockerfile ../../..
```

### 2. Prepare Your Model

Place your trained model in a directory that will be mounted to the container:

```bash
mkdir -p ./models
cp /path/to/your/model.pt ./models/model.pt
```

### 3. Run the Container

```bash
docker run -d \
  --name triton-inference \
  -p 5000:5000 \
  -v $(pwd)/models:/app/saved_models \
  -e MODEL_PATH=/app/saved_models/model.pt \
  triton-inference:latest
```

### 4. Test the API

```bash
# Health check
curl http://localhost:5000/health

# Model info
curl http://localhost:5000/info

# Make a prediction
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/predict
```

## API Endpoints

### Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Response**: Server health status

```bash
curl http://localhost:5000/health
```

### Model Information
- **URL**: `/info`
- **Method**: `GET`
- **Response**: Model metadata and statistics

```bash
curl http://localhost:5000/info
```

### Single Image Prediction
- **URL**: `/predict`
- **Method**: `POST`
- **Parameters**:
  - `image` (file): Image file to classify
  - `top_k` (optional, int): Number of top predictions (default: 5)
  - `transform` (optional, string): Transform type ('cifar' or 'imagenet')

```bash
curl -X POST \
  -F "image=@image.jpg" \
  -F "top_k=3" \
  http://localhost:5000/predict
```

**Response**:
```json
{
  "predictions": [
    {
      "class_id": 3,
      "class_name": "cat",
      "probability": 0.8234,
      "confidence": 82.34
    },
    {
      "class_id": 5,
      "class_name": "dog",
      "probability": 0.1245,
      "confidence": 12.45
    }
  ],
  "inference_time_ms": 12.34,
  "total_time_ms": 15.67,
  "device": "cpu"
}
```

### Batch Prediction
- **URL**: `/predict_batch`
- **Method**: `POST`
- **Parameters**:
  - `images` (multiple files): Multiple image files
  - `top_k` (optional, int): Number of top predictions

```bash
curl -X POST \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "images=@image3.jpg" \
  http://localhost:5000/predict_batch
```

### Metrics (Prometheus)
- **URL**: `/metrics`
- **Method**: `GET`
- **Response**: Prometheus-format metrics

```bash
curl http://localhost:5000/metrics
```

## Configuration Options

### Environment Variables

- `MODEL_PATH`: Path to model file (default: `/app/saved_models/model.pt`)
- `CLASS_NAMES_PATH`: Path to class names JSON (optional)
- `NUM_WORKERS`: Number of worker processes (default: 4)
- `PORT`: Server port (default: 5000)
- `DEBUG`: Enable debug mode (default: false)

### Example with Custom Configuration

```bash
docker run -d \
  --name triton-inference \
  -p 8080:8080 \
  -v $(pwd)/models:/app/saved_models \
  -e MODEL_PATH=/app/saved_models/custom_model.pt \
  -e PORT=8080 \
  -e NUM_WORKERS=8 \
  triton-inference:latest
```

## Production Deployment

### Using Gunicorn

For production, use Gunicorn instead of the Flask development server:

```bash
docker run -d \
  --name triton-inference \
  -p 5000:5000 \
  -v $(pwd)/models:/app/saved_models \
  triton-inference:latest \
  gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Compose

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  inference:
    build:
      context: ../../..
      dockerfile: examples/deployment/docker_deployment/Dockerfile
    image: triton-inference:latest
    container_name: triton-inference
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/saved_models
    environment:
      - MODEL_PATH=/app/saved_models/model.pt
      - NUM_WORKERS=4
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 10s
```

Run with:

```bash
docker-compose up -d
```

## Scaling and Load Balancing

### Horizontal Scaling

Run multiple instances:

```bash
docker run -d --name triton-1 -p 5001:5000 triton-inference:latest
docker run -d --name triton-2 -p 5002:5000 triton-inference:latest
docker run -d --name triton-3 -p 5003:5000 triton-inference:latest
```

### With Nginx Load Balancer

Create `nginx.conf`:

```nginx
upstream triton_backend {
    least_conn;
    server localhost:5001;
    server localhost:5002;
    server localhost:5003;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://triton_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 10M;
    }
}
```

## GPU Support

To use GPU acceleration:

### 1. Install NVIDIA Container Toolkit

```bash
# Ubuntu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Run with GPU

```bash
docker run -d \
  --name triton-inference-gpu \
  --gpus all \
  -p 5000:5000 \
  -v $(pwd)/models:/app/saved_models \
  triton-inference:latest
```

## Monitoring

### Prometheus Integration

Add Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: 'triton-inference'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/metrics'
```

### Grafana Dashboard

Import metrics:
- `inference_requests_total`: Total number of requests
- `inference_request_duration_seconds`: Request latency
- `model_load_time_seconds`: Model loading time
- `active_inference_requests`: Active requests

## Testing

### Python Client Example

```python
import requests

# Single prediction
with open('image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/predict', files=files)
    print(response.json())

# Batch prediction
files = [
    ('images', open('image1.jpg', 'rb')),
    ('images', open('image2.jpg', 'rb')),
    ('images', open('image3.jpg', 'rb'))
]
response = requests.post('http://localhost:5000/predict_batch', files=files)
print(response.json())
```

### cURL Examples

```bash
# Health check
curl http://localhost:5000/health

# Single image
curl -X POST -F "image=@cat.jpg" http://localhost:5000/predict

# Multiple images
curl -X POST \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  http://localhost:5000/predict_batch

# With parameters
curl -X POST \
  -F "image=@image.jpg" \
  -F "top_k=10" \
  -F "transform=imagenet" \
  http://localhost:5000/predict
```

## Troubleshooting

### Check Container Logs

```bash
docker logs triton-inference
```

### Interactive Shell

```bash
docker exec -it triton-inference /bin/bash
```

### Test Model Loading

```bash
docker exec triton-inference python -c "import torch; model = torch.load('/app/saved_models/model.pt'); print('Model loaded')"
```

### Memory Issues

If running out of memory:

```bash
# Limit memory usage
docker run -d \
  --name triton-inference \
  --memory="2g" \
  --memory-swap="2g" \
  -p 5000:5000 \
  triton-inference:latest
```

### Port Already in Use

```bash
# Use different port
docker run -d -p 8080:5000 triton-inference:latest
```

## Security Considerations

### 1. Non-root User

Modify Dockerfile to run as non-root:

```dockerfile
RUN useradd -m -u 1000 triton && chown -R triton:triton /app
USER triton
```

### 2. Read-only Filesystem

```bash
docker run -d \
  --read-only \
  --tmpfs /tmp \
  -v $(pwd)/models:/app/saved_models:ro \
  triton-inference:latest
```

### 3. Resource Limits

```bash
docker run -d \
  --cpus="2" \
  --memory="4g" \
  --pids-limit=100 \
  triton-inference:latest
```

## Performance Optimization

### 1. Optimize Docker Image Size

- Use multi-stage builds (already implemented)
- Remove unnecessary dependencies
- Use `.dockerignore` file

### 2. Model Optimization

- Use TorchScript for faster inference
- Apply quantization
- Batch requests when possible

### 3. Caching

Enable Docker BuildKit for better caching:

```bash
DOCKER_BUILDKIT=1 docker build -t triton-inference:latest .
```

## Kubernetes Deployment

Example Kubernetes manifest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: triton-inference
  template:
    metadata:
      labels:
        app: triton-inference
    spec:
      containers:
      - name: triton
        image: triton-inference:latest
        ports:
        - containerPort: 5000
        env:
        - name: MODEL_PATH
          value: /app/saved_models/model.pt
        volumeMounts:
        - name: models
          mountPath: /app/saved_models
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: triton-inference
spec:
  selector:
    app: triton-inference
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/financecommander/Triton/issues
- Documentation: https://github.com/financecommander/Triton/docs

## License

MIT License - see LICENSE file for details
