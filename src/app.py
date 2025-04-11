import os
import json
from flask import Flask, request, jsonify, render_template
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

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

def load_jsonl_file(file_path):
    """
    Charge manuellement un fichier JSONL et le convertit en documents LangChain
    """
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                try:
                    line = line.strip()
                    if not line:  # Ignore les lignes vides
                        continue
                        
                    data = json.loads(line)
                    
                    # Essayer d'extraire le contenu textuel du document
                    content = None
                    
                    # Chercher des clés courantes pour le contenu
                    for key in ['text', 'content', 'page_content', 'body', 'document']:
                        if key in data and isinstance(data[key], str):
                            content = data[key]
                            break
                    
                    # Si aucune clé standard n'est trouvée, essayer de trouver une chaîne de caractères longue
                    if content is None:
                        for key, value in data.items():
                            if isinstance(value, str) and len(value) > 50:
                                content = value
                                break
                    
                    # En dernier recours, utiliser tout le JSON comme contenu
                    if content is None:
                        content = json.dumps(data)
                    
                    # Créer un Document LangChain
                    metadata = {"source": file_path, "line": line_number}
                    # Ajouter d'autres métadonnées si disponibles
                    for meta_key in ['title', 'url', 'author', 'date', 'source']:
                        if meta_key in data and isinstance(data[meta_key], (str, int, float, bool)):
                            metadata[meta_key] = data[meta_key]
                    
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
                    
                except json.JSONDecodeError as e:
                    print(f"Erreur JSON à la ligne {line_number}: {e}")
                except Exception as e:
                    print(f"Erreur lors du traitement de la ligne {line_number}: {e}")
    except Exception as e:
        print(f"Erreur lors de l'ouverture du fichier {file_path}: {e}")
    
    return documents

def initialize_rag_system():
    global vectorstore, rag_chain
    
    print("Initializing RAG system...")
    
    # Définir le chemin vers le fichier train.jsonl téléchargé
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    jsonl_file = os.path.join(data_dir, 'train.jsonl')
    
    # Vérifier si le fichier existe
    if not os.path.exists(jsonl_file):
        print(f"ATTENTION: Le fichier {jsonl_file} n'existe pas.")
        print("Veuillez télécharger ce fichier depuis Hugging Face et le placer dans le dossier 'data'.")
        
        # Créer un fichier d'exemple si le fichier n'existe pas
        sample_file = os.path.join(data_dir, 'langchain_sample.txt')
        if not os.path.exists(sample_file):
            with open(sample_file, 'w', encoding='utf-8') as f:
                f.write("""
# LangChain Documentation

LangChain is a framework for developing applications powered by language models. It enables applications that:
- Are context-aware: connect a language model to sources of context
- Reason: rely on a language model to reason about provided context

## Key Components

### Chains
Chains in LangChain go beyond a single LLM call and involve sequences of calls.

### Agents
Agents involve an LLM making decisions about which actions to take.

### Retrievers
Retrievers retrieve documents from a source. LangChain provides a standard interface for retrievers.
                """)
            documents = [Document(page_content=open(sample_file, 'r', encoding='utf-8').read(), metadata={"source": sample_file})]
            print(f"Créé un fichier d'exemple à la place: {sample_file}")
        else:
            documents = [Document(page_content=open(sample_file, 'r', encoding='utf-8').read(), metadata={"source": sample_file})]
            print(f"Utilisation du fichier d'exemple existant: {sample_file}")
    else:
        # Charger les documents depuis le fichier JSONL
        print("Chargement des documents depuis le fichier JSONL...")
        documents = load_jsonl_file(jsonl_file)
        print(f"Chargés {len(documents)} documents depuis {jsonl_file}")
    
    if len(documents) == 0:
        print("ERREUR: Aucun document n'a été chargé. Impossible de continuer.")
        return None
    
    # Traiter les documents
    print("Traitement des documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Créés {len(chunks)} fragments de documents")
    
    if len(chunks) == 0:
        print("ERREUR: Aucun fragment de document n'a été créé. Impossible de continuer.")
        return None
    
    # Configuration des embeddings et du vectorstore
    print("Configuration des embeddings et de la base vectorielle...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    try:
        # Pour supprimer l'avertissement de dépréciation, utiliser la nouvelle importation si disponible
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
        except ImportError:
            embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
        
        persist_directory = os.path.join(data_dir, "chroma_db")
        os.makedirs(persist_directory, exist_ok=True)
        
        # Créer la base vectorielle
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        
    except Exception as e:
        print(f"Erreur lors de la configuration des embeddings ou de la base vectorielle: {str(e)}")
        return None
    
    # Configuration du pipeline RAG
    print("Configuration du pipeline RAG...")
    
    # Initialiser le modèle de langage depuis Ollama
    try:
        llm = Ollama(model="llama2")
    except Exception as e:
        print(f"Erreur lors de l'initialisation d'Ollama: {str(e)}")
        print("Assurez-vous qu'Ollama est installé et en cours d'exécution")
        return None
        
    # Créer un récupérateur à partir de la base vectorielle
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Définir le template pour le prompt RAG
    template = """
    Vous êtes un assistant utile qui répond aux questions sur LangChain en vous basant sur le contexte fourni.
    Utilisez les extraits suivants pour répondre à la question posée à la fin.
    Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse.
    Gardez vos réponses concises et concentrées sur la question.

    Contexte:
    {context}

    Question: {question}

    Votre réponse:
    """
    
    # Créer le prompt à partir du template
    prompt = PromptTemplate.from_template(template)
    
    # Définir une fonction pour formater les documents pour le contexte
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    # Construire la chaîne RAG
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("Système RAG initialisé avec succès!")
    return rag_chain

def run_app():
    global rag_chain
    
    # Initialiser le système RAG
    rag_chain = initialize_rag_system()
    
    if rag_chain is None:
        print("Échec de l'initialisation du système RAG")
        return
    
    # Exécuter l'application Flask
    app.run(debug=True, host='0.0.0.0', port=5000)