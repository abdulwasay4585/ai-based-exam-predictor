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

# Create user first for correct permissions
RUN useradd -m -u 1000 user

# Install system dependencies as root
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Switch to the non-root user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Copy requirements
COPY --chown=user backend/requirements.txt ./backend/requirements.txt
COPY --chown=user ml_pipeline/requirements.txt ./ml_pipeline/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r ./ml_pipeline/requirements.txt
RUN pip install --no-cache-dir -r ./backend/requirements.txt

# Explicitly download the spacy model to avoid runtime errors
RUN python -m spacy download en_core_web_sm

# Copy backend and ML pipeline code
COPY --chown=user . $HOME/app

# Copy built React files from Stage 1 into the backend's static directory OVERWRITING the local empty static dir if it exists
COPY --from=frontend-builder --chown=user /app/frontend/build $HOME/app/backend/app/static

# Start the FastAPI server using Uvicorn on Hugging Face's required port (7860)
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
