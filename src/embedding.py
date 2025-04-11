import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def setup_embeddings(documents, persist_directory=None):
    """
    Set up the embedding model and vector store
    
    Args:
        documents: List of document chunks to embed
        persist_directory: Directory to persist the vector store (optional)
        
    Returns:
        embedding_model: The embedding model
        vectorstore: The Chroma vector store
    """
    # Configure embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # Set up persist directory if provided
    if persist_directory is None:
        # Create a default directory in the project's data folder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        persist_directory = os.path.join(base_dir, "data", "chroma_db")
    
    # Ensure the directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    # Persist the database
    vectorstore.persist()
    
    return embedding_model, vectorstore