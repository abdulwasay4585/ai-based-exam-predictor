---
title: AI Exam Predictor
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# AI Exam Question Prediction System

This repository contains the full stack implementation of the AI Exam Prediction platform. The system leverages advanced natural language processing and machine learning to analyze academic materials and predict high probability exam questions based on historical trends.

## Architecture

The project is divided into distinct modular components.

* backend directory: FastAPI serving endpoints and orchestration logic.
* frontend directory: React user interface with Markdown and LaTeX rendering capabilities.
* ml pipeline directory: PyTorch models, NLP components, and training scripts.
* data directory: Datasets and raw notes.
* saved models directory: Trained model weights.

## Core Technologies

* Frontend: React, Tailwind CSS, Framer Motion, KaTeX
* Backend: FastAPI, Python
* Machine Learning: PyTorch, Transformers, FAISS, SpaCy, scikit learn
* Deployment: Docker

## Key Features

* Document Analysis: Extracts text from PDF files and performs intelligent chunking.
* Topic Modeling: Classifies and identifies core academic themes from uploaded notes.
* Trend Prediction: Analyzes historical exam frequencies to predict future test topics.
* Question Generation: Generates highly relevant exam questions using Retrieval Augmented Generation.
* Automated Solving: Provides AI generated solutions to predicted questions.
* Unified Deployment: Designed to be easily hosted using a single Docker container.

## Local Development Setup

To run the project locally, you will need Node and Python installed on your system.

### Frontend Setup

1. Change directory to the frontend folder.
2. Run npm install to download all necessary dependencies.
3. Run npm start to launch the development server on localhost port 3000.

### Backend Setup

1. Change directory to the backend folder.
2. Install the required Python packages using pip install -r requirements.txt.
3. Start the server using uvicorn app.main:app --host 0.0.0.0 --port 8000.

### Machine Learning Pipeline

The ml pipeline folder contains all the heavy computation scripts.
Ensure you install the dependencies located in the ml pipeline requirements.txt file before running any model training or evaluation scripts.

## Deployment

A unified Dockerfile is located in the root directory. This configuration builds the React frontend, sets up the Python backend environment, and configures FastAPI to serve the static frontend files. This allows the entire application to run seamlessly on services like Hugging Face Spaces.
