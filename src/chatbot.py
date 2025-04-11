def generate_response(rag_chain, query):
    """
    Generate a response using the RAG chain
    
    Args:
        rag_chain: The RAG chain
        query: The user query
        
    Returns:
        response: The generated response
    """
    try:
        # Invoke the RAG chain with the query
        response = rag_chain.invoke(query)
        return response
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return f"An error occurred while generating a response: {str(e)}"