import os
import ollama
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Configuration (matching RAG_Tutorial.ipynb Step 2)
DATA_FOLDER = Path("../data")
DB_FOLDER = Path("../chroma_db")
CHUNK_SIZE = 500  # words per chunk
CHUNK_OVERLAP = 100  # overlapping words
MODEL_NAME = "mistral"  # Your Ollama model
COLLECTION_NAME = "ncert_documents"

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Ensure data folder exists
DATA_FOLDER.mkdir(exist_ok=True)

# Global variables
embedding_model = None
chroma_client = None
collection = None

print(f"📁 Data folder: {DATA_FOLDER.absolute()}")
print(f"💾 Database folder: {DB_FOLDER.absolute()}")
print(f"🤖 LLM Model: {MODEL_NAME}")


def initialize_models():
    """Initialize embedding model and ChromaDB (Steps 3 & 4)"""
    global embedding_model, chroma_client, collection
    
    # Load embedding model (Step 3)
    print("Loading embedding model... (this may take a minute first time)")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Embedding model loaded!")
    print(f"   Model dimension: {embedding_model.get_sentence_embedding_dimension()}")
    
    # Initialize ChromaDB (Step 4)
    print("Initializing ChromaDB...")
    chroma_client = chromadb.PersistentClient(
        path=str(DB_FOLDER),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Get or create collection
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "NCERT PDFs and study notes"}
    )
    
    print("✅ ChromaDB initialized!")
    print(f"   Current documents in DB: {collection.count()}")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF file (Step 5)"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page_num, page in enumerate(reader.pages, 1):
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"❌ Error reading {pdf_path.name}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks (Step 6)"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


def retrieve_relevant_chunks(query: str, n_results: int = 5) -> Dict:
    """Retrieve relevant chunks for a query (Step 9)"""
    # Generate query embedding
    query_embedding = embedding_model.encode([query])[0]
    
    # Search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    
    return results


# Routes

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "message": "RAG Backend is running",
        "model": MODEL_NAME,
        "documents_count": collection.count() if collection else 0
    })


@app.route('/upload', methods=['POST'])
def upload_document():
    """
    Upload a PDF document (Step 8)
    Request: multipart/form-data with 'file' field
    """
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = DATA_FOLDER / filename
        file.save(str(filepath))
        
        print(f"📄 Processing: {filename}")
        
        # Extract text (Step 5)
        text = extract_text_from_pdf(filepath)
        if not text.strip():
            return jsonify({"error": "No text extracted from PDF"}), 400
        
        word_count = len(text.split())
        print(f"   ✓ Extracted {word_count:,} words")
        
        # Chunk text (Step 6)
        chunks = chunk_text(text)
        print(f"   ✓ Created {len(chunks)} chunks")
        
        # Prepare data for ChromaDB (Step 8)
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": filename,
                "chunk_id": idx,
                "total_chunks": len(chunks)
            })
            all_ids.append(f"{Path(filename).stem}_{idx}")
        
        # Generate embeddings (Step 8)
        print(f"🧠 Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = embedding_model.encode(all_chunks, show_progress_bar=True)
        
        # Upload to ChromaDB (Step 8)
        print("💾 Uploading to ChromaDB...")
        collection.add(
            embeddings=embeddings.tolist(),
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        print(f"✅ Success! Uploaded {len(all_chunks)} chunks")
        print(f"   Total documents in DB: {collection.count()}")
        
        return jsonify({
            "message": "Document uploaded and processed successfully",
            "fileName": filename,
            "chunks": len(chunks),
            "words": word_count
        })
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Ask a question using RAG (Step 11)
    Request: {"question": "Your question here"}
    Response: {"answer": "...", "sources": [...]}
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({"error": "Question is required"}), 400
        
        question = data['question']
        print(f"❓ Question: {question}")
        
        # Check if database has content
        if collection.count() == 0:
            return jsonify({
                "error": "Database is empty. Please upload PDFs first!"
            }), 400
        
        print("🔍 Retrieving relevant content...")
        
        # Retrieve relevant chunks (Step 9)
        results = retrieve_relevant_chunks(question, n_results=5)
        
        if not results['documents'] or not results['documents'][0]:
            return jsonify({
                "answer": "No relevant documents found.",
                "sources": []
            })
        
        # Prepare context (Step 11)
        context = "\n\n".join(results['documents'][0])
        metadatas = results['metadatas'][0]
        sources = set(meta['source'] for meta in metadatas)
        print(f"✓ Found content from: {', '.join(sources)}")
        
        # Create prompt (Step 11)
        prompt = f"""Based on the following context from NCERT and study materials, answer the question accurately.

Context:
{context}

Question: {question}

Answer the question based only on the provided context. If the context doesn't contain enough information, say so."""
        
        # Query Ollama (Step 11)
        print(f"🤖 Generating answer using {MODEL_NAME}...")
        
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{'role': 'user', 'content': prompt}],
                stream=False
            )
            
            answer = response['message']['content']
            print("✅ Answer generated")
            
            # Format sources
            formatted_sources = [
                {
                    "file": meta['source'],
                    "chunk": meta['chunk_id']
                }
                for meta in metadatas
            ]
            
            return jsonify({
                "answer": answer,
                "sources": formatted_sources
            })
            
        except Exception as ollama_error:
            print(f"❌ Ollama error: {ollama_error}")
            return jsonify({
                "error": "Failed to generate answer. Make sure Ollama is running.",
                "details": f"Run 'ollama serve' and 'ollama pull {MODEL_NAME}'"
            }), 500
        
    except Exception as e:
        print(f"❌ Ask error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/sources', methods=['GET'])
def get_sources():
    """
    Get list of all uploaded documents (Step 13)
    Response: {"sources": [...], "totalChunks": ...}
    """
    try:
        count = collection.count()
        
        if count == 0:
            return jsonify({"sources": [], "totalChunks": 0})
        
        # Get all documents from collection
        results = collection.get()
        
        if not results['metadatas']:
            return jsonify({"sources": [], "totalChunks": 0})
        
        # Extract unique files and count chunks (Step 13)
        files_map = {}
        for meta in results['metadatas']:
            source = meta['source']
            if source not in files_map:
                files_map[source] = {
                    "file": source,
                    "chunks": 0
                }
            files_map[source]["chunks"] += 1
        
        sources = list(files_map.values())
        
        return jsonify({
            "sources": sources,
            "totalChunks": count
        })
        
    except Exception as e:
        print(f"❌ Get sources error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Initialize models on startup
    initialize_models()
    
    # Run Flask app
    PORT = int(os.environ.get('PORT', 3000))
    print(f"\n🚀 Starting Flask server on http://localhost:{PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=True)
