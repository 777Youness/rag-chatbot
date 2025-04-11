import os
from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from src.embedding import setup_embeddings
from src.rag import setup_rag_pipeline
from src.chatbot import generate_response
from src.utils import process_documents

# Initialize Flask app
app = Flask(__name__, 
            template_folder='web/templates',
            static_folder='web/static')

# Global variables
vectorstore = None
rag_chain = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global rag_chain
    
    if rag_chain is None:
        return jsonify({'error': 'RAG system not initialized'}), 500
        
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    try:
        response = generate_response(rag_chain, query)
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return jsonify({'error': f'Failed to generate response: {str(e)}'}), 500

def initialize_rag_system():
    global vectorstore, rag_chain
    
    print("Initializing RAG system...")
    
    # Check if data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'langchain_docs')
    os.makedirs(data_dir, exist_ok=True)
    
    # Load dataset
    print("Loading documents...")
    try:
        loader = HuggingFaceDatasetLoader(
            repo_id="antonioibars/langchain-docs",
            split="train"
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        documents = []

    # Process documents
    print("Processing documents...")
    chunks = process_documents(documents)
    print(f"Created {len(chunks)} document chunks")
    
    # Setup embeddings and vectorstore
    print("Setting up embeddings and vector store...")
    embedding_model, vectorstore = setup_embeddings(chunks)
    
    # Setup RAG pipeline
    print("Setting up RAG pipeline...")
    rag_chain = setup_rag_pipeline(embedding_model, vectorstore)
    
    print("RAG system initialized successfully!")
    return rag_chain

if __name__ == '__main__':
    # Initialize the RAG system
    rag_chain = initialize_rag_system()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)