# LangChain RAG Chatbot

A Retrieval Augmented Generation (RAG) chatbot that answers questions about LangChain documentation.

## Features

- Responds to natural language queries about LangChain
- Uses RAG architecture to provide accurate, context-aware responses
- Web-based interface for easy interaction
- Powered by LangChain, Ollama, and ChromaDB

## Architecture

This project implements a RAG (Retrieval Augmented Generation) system with the following components:

1. **Document Processing**: Loads LangChain documentation from Hugging Face and splits it into chunks
2. **Embedding**: Uses `sentence-transformers/all-MiniLM-L6-v2` to create embeddings
3. **Vector Storage**: ChromaDB stores document embeddings for efficient retrieval
4. **Retrieval**: When a query is received, finds the most relevant document chunks
5. **Generation**: Uses Ollama with a carefully crafted prompt to generate responses
6. **Web Interface**: HTML/CSS/JS frontend for easy interaction

## Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running

### Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd rag-chatbot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download and run a model with Ollama:
   ```
   # Download and run llama2
   ollama pull llama2
   # Verify Ollama is running
   ollama run llama2 "Hello, world!"
   ```

## Usage

1. Start the application:
   ```
   python run.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Ask questions about LangChain in the chat interface!

## Configuration

- The default Ollama model is set to `llama2`. You can change this in `src/rag.py`.
- Adjust chunk size and overlap in `src/utils.py` if needed.
- The number of retrieved documents is set to 4 in `src/rag.py`.

## Project Structure

```
rag-chatbot/
├── README.md
├── requirements.txt
├── RAPPORT.md
├── data/
│   └── langchain_docs/
├── src/
│   ├── main.py
│   ├── embedding.py
│   ├── rag.py
│   ├── chatbot.py
│   ├── utils.py
│   └── web/
│       ├── static/
│       │   ├── css/
│       │   │   └── style.css
│       │   └── js/
│       │       └── script.js
│       └── templates/
│           └── index.html
└── tests/
    └── test_queries.py
```

## Example Queries

- "How do I use LangChain with OpenAI models?"
- "Explain the concept of chains in LangChain"
- "What are embeddings and how are they used in LangChain?"
- "How do I implement a simple chatbot with LangChain?"
- "What is the difference between a Chain and an Agent in LangChain?"

## Troubleshooting

- If you encounter issues with Ollama, make sure the Ollama service is running
- If embeddings fail, check that you have sufficient disk space for the models
- For other issues, check the console output for error messages