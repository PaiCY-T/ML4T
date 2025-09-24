# Production Dockerfile for ML4T LightGBM Inference System
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r ml4t && useradd -r -g ml4t ml4t

# Copy requirements first for better caching
COPY requirements.txt .
COPY deployment/requirements-production.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-production.txt

# Copy application code
COPY src/ ./src/
COPY deployment/production_server.py .
COPY deployment/health_check.py .

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/data && \
    chown -R ml4t:ml4t /app

# Switch to non-root user
USER ml4t

# Expose ports
EXPOSE 8080 8090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python health_check.py

# Start the production server
CMD ["python", "production_server.py"]