from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def setup_rag_pipeline(embedding_model, vectorstore, model_name="llama2"):
    """
    Set up the RAG pipeline
    
    Args:
        embedding_model: The embedding model
        vectorstore: The vector store containing embedded documents
        model_name: The Ollama model to use
        
    Returns:
        rag_chain: The RAG chain
    """
    # Initialize the language model from Ollama
    llm = Ollama(model=model_name)
    
    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Retrieve 4 most similar documents
    )
    
    # Define the template for the RAG prompt
    template = """
    You are a helpful assistant that answers questions about LangChain based on the provided context.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep your answers concise and focused on the question.

    Context:
    {context}

    Question: {question}

    Your Answer:
    """
    
    # Create the prompt from the template
    prompt = PromptTemplate.from_template(template)
    
    # Define a function to format the documents for the context
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    # Build the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain