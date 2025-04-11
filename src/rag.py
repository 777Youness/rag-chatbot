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
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.6}  # Augmenter le seuil
    )
    
    
    # Define the template for the RAG prompt
    template = """
    You are a specialized assistant that ONLY answers questions about LangChain documentation.
    Use ONLY the following context to answer the question. 
    If the question is not about LangChain, or if the context doesn't contain the information needed, 
    respond with: "Je ne peux répondre qu'à des questions sur LangChain basées sur la documentation fournie."
    Do not make up information or use your general knowledge.

    Context:
    {context}

    Question: {question}

    Your Answer (ONLY about LangChain based on the context above):
    """
    
    # Create the prompt from the template
    prompt = PromptTemplate.from_template(template)
    
    # Define a function to format the documents for the context
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    
    def print_debug_info(inputs):
        print(f"QUESTION: {inputs['question']}")
        print(f"CONTEXT CHUNKS: {len(inputs['context'].split('\\n\\n'))}")
        print(f"CONTEXT SAMPLE: {inputs['context'][:300]}...")
        return inputs
    
    # Build the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


