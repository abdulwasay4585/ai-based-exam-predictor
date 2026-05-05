# AI-Based Exam Predictor

## Overview
The AI-Based Exam Predictor is a sophisticated platform designed to analyze academic materials and predict potential exam questions. Using state-of-the-art Natural Language Processing (NLP) techniques and Retrieval-Augmented Generation (RAG), the system provides students and educators with insights into key topics and trends within their curricula.

## Project Structure
The repository is organized into several decoupled services:

- **backend/**: A high-performance REST API built with FastAPI, handling data retrieval, user requests, and model orchestration.
- **frontend/**: A modern user interface developed with React, offering a responsive and intuitive experience for interacting with the prediction engine.
- **ml_pipeline/**: The core intelligence of the system, containing PyTorch model architectures, NLP preprocessing scripts, and training loops.
- **notebooks/**: Exploratory Data Analysis (EDA) and model prototyping notebooks.
- **scripts/**: Utility scripts for data management and deployment.

## Features
- **Trend Analysis**: Identification of high-frequency topics from historical data and course materials.
- **Question Generation**: Automatic generation of relevant exam questions based on specific input documents.
- **Topic Modeling**: Clustering and categorization of academic content to highlight core concepts.
- **Containerized Architecture**: Full support for Docker and Docker Compose for seamless deployment across environments.

## Getting Started

### Prerequisites
- Docker and Docker Compose
- Python 3.9+
- Node.js (for frontend development)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/abdulwasay4585/ai-based-exam-predictor.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ai-based-exam-predictor
   ```
3. Start the services using Docker Compose:
   ```bash
   docker-compose up --build
   ```

## Development
To run services individually for development purposes, refer to the documentation within the `backend/` and `frontend/` directories.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
