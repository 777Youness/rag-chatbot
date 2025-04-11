import os
from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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

def generate_response(rag_chain, query):
    """Generate a response using the RAG chain"""
    try:
        response = rag_chain.invoke(query)
        return response
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return f"An error occurred while generating a response: {str(e)}"

def initialize_rag_system():
    global vectorstore, rag_chain
    
    print("Initializing RAG system...")
    
    # Create sample data directory and file if they don't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    sample_file = os.path.join(data_dir, 'langchain_sample.txt')
    
    # Create sample content if file doesn't exist
    if not os.path.exists(sample_file):
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write("""
# LangChain Documentation

LangChain is a framework for developing applications powered by language models. It enables applications that:
- Are context-aware: connect a language model to sources of context (prompt instructions, few-shot examples, content to ground its response in, etc.)
- Reason: rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.)

## Modules

LangChain provides modules that help with:
- Model I/O: Interface with language models
- Retrieval: Interface with application-specific data
- Agents: Let chains choose which tools to use given high-level directives

## Key Concepts

### Chains
Chains go beyond a single LLM call and involve sequences of calls (whether to an LLM or a different utility). LangChain provides a standard interface for chains, and several implementations of chains for different use cases.

### Agents
Agents involve an LLM making decisions about which actions to take, taking that action, seeing an observation, and repeating until done. LangChain provides a standard interface for agents, and several implementations of agents.

### Memory
Memory refers to persisting state between calls of a chain/agent. LangChain provides a standard interface for memory, and several implementations of memory for different use cases.

### Callbacks
Callbacks allow you to log and stream intermediate steps of any chain, making it easy to observe, debug, and evaluate the internals of an application.

### Document Loaders
Document loaders load documents from many different sources. LangChain provides a standard interface for document loaders, and several implementations.

### Text Splitters
Text splitters split documents into chunks. LangChain provides several implementations of text splitters.

### Retrievers
Retrievers retrieve documents from a source. LangChain provides a standard interface for retrievers, and several implementations.

### Embeddings
Embedding models convert text into numerical representations. LangChain integrates with various embedding models.

### Vector Stores
Vector stores store embeddings and allow for searching based on similarity. LangChain provides a standard interface for vector stores, and several implementations.
            """)
    
    # Load documents
    print("Loading documents...")
    try:
        loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        documents = []

    # Process documents
    print("Processing documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} document chunks")
    
    # Setup embeddings and vectorstore
    print("Setting up embeddings and vector store...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    persist_directory = os.path.join(data_dir, "chroma_db")
    os.makedirs(persist_directory, exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    # Setup RAG pipeline
    print("Setting up RAG pipeline...")
    
    # Initialize the language model from Ollama
    try:
        llm = Ollama(model="llama2")
    except Exception as e:
        print(f"Error initializing Ollama: {str(e)}")
        print("Make sure Ollama is installed and running")
        return None
        
    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
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
    
    print("RAG system initialized successfully!")
    return rag_chain

def run_app():
    global rag_chain
    
    # Initialize the RAG system
    rag_chain = initialize_rag_system()
    
    if rag_chain is None:
        print("Failed to initialize RAG system")
        return
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)