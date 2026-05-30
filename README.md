# Python Docs Copilot

A RAG system designed to answer questions about Python based on the official Python documentation.

The project is implemented in Python and uses FAISS, Hugging Face Transformers, Sentence Transformers, LangChain, and Gradio.

The application is containerized with Docker Compose and currently consists of two services:

1. **UI service** — a simple web interface for asking questions
2. **RAG pipeline service** — performs retrieval, reranking, validation, and answer generation

A third service for downloading the latest Python documentation, performing chunking, and building the FAISS index is planned for future development.

## Prerequisites

Before running the project, make sure the following tools are installed:

* Docker
* Docker Compose
* Git
* NVIDIA Container Runtime

## Building the project

Clone the repository:

```bash
git clone https://github.com/BognaLew/Python_Docs_Copilot.git
cd Python_Docs_Copilot
```

Build all containers:

```bash
docker compose build
```

## Running the application

Start all services:

```bash
docker compose up
```

The application will be available at:

```text
http://localhost:8080
```

## Planned improvements

* Automatic Python documentation downloader/updater service
* Support for other Python libraries
* Conversation memory
* Improved UI/UX
