from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_documents(documents, chunk_size=512, chunk_overlap=100):
    """
    Process documents by splitting them into chunks
    
    Args:
        documents: List of documents to process
        chunk_size: Size of each chunk in tokens
        chunk_overlap: Amount of overlap between chunks in tokens
        
    Returns:
        chunks: List of document chunks
    """
    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Split the documents into chunks
    chunks = text_splitter.split_documents(documents)
    
    return chunks