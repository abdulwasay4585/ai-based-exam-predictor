# ==========================================
# Stage 1: Build the React Frontend
# ==========================================
FROM node:18 AS frontend-builder
WORKDIR /app/frontend

# Install dependencies
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install

# Copy the rest of the frontend code and build it
COPY frontend/ ./
RUN npm run build

# ==========================================
# Stage 2: Set up FastAPI Backend + PyTorch
# ==========================================
FROM python:3.10-slim

# Set huggingface spaces default port
ENV PORT=7860
ENV HOST=0.0.0.0

WORKDIR /app

# Install system dependencies (needed for PyMuPDF, faiss, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and ML pipeline requirements
COPY backend/requirements.txt /app/backend/requirements.txt
COPY ml_pipeline/requirements.txt /app/ml_pipeline/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/ml_pipeline/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend and ML pipeline code
COPY backend /app/backend
COPY ml_pipeline /app/ml_pipeline

# Copy built React files from Stage 1 into the backend's static directory
COPY --from=frontend-builder /app/frontend/build /app/backend/app/static

# Give user permissions (HuggingFace spaces requires non-root user for security)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Start the FastAPI server using Uvicorn on Hugging Face's required port (7860)
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
