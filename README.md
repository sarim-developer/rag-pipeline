# NCERT RAG System 📚

A Retrieval-Augmented Generation (RAG) system for querying NCERT PDFs and study notes using ChromaDB and Ollama.

## Architecture

```
PDFs (data/)
    ↓
PDF Parser (pypdf)
    ↓
Chunking (500 words with 50 word overlap)
    ↓
Embeddings (sentence-transformers: all-MiniLM-L6-v2)
    ↓
Vector Database (ChromaDB)
    ↓
Retriever (semantic search)
    ↓
Local LLM (Ollama)
    ↓
Answer + Sources
```

## Features

- 📄 **PDF Parsing**: Extracts text from multiple PDF files
- ✂️ **Smart Chunking**: Splits text into overlapping chunks for better context
- 🧠 **Embeddings**: Uses sentence-transformers for semantic understanding
- 💾 **Vector DB**: Persistent storage with ChromaDB
- 🔍 **Semantic Search**: Retrieves most relevant chunks for any query
- 🤖 **Local LLM**: Uses Ollama for privacy-focused answer generation
- 📚 **Source Tracking**: Always shows which PDFs were used for answers

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama

**macOS:**
```bash
brew install ollama
```

Or download from: https://ollama.ai/download

### 3. Start Ollama Service

```bash
ollama serve
```

### 4. Pull a Model

```bash
# Recommended: Llama 3.2 (smaller, faster)
ollama pull llama3.2

# Or for better quality (larger):
ollama pull llama3.1
```

## Usage

### 1. Add Your PDFs

Create a `data` folder and add your NCERT PDFs:

```bash
mkdir data
# Copy your PDFs to the data folder
```

### 2. Run the System

```bash
python main.py
```

### 3. Interactive Menu

```
1. Upload PDFs from data folder    - Process and index all PDFs
2. Ask a question                  - Query your documents
3. View database statistics        - See what's indexed
4. Clear database                  - Start fresh
5. Exit                           - Quit the application
```

## Example Usage

```
📚 NCERT RAG System

Options:
  1. Upload PDFs from data folder
  2. Ask a question
  3. View database statistics
  4. Clear database
  5. Exit

Enter your choice: 1

Found 3 PDF files. Processing...

Processing: history_ncert.pdf
  ✓ Extracted 245 chunks

Processing: polity_notes.pdf
  ✓ Extracted 178 chunks

✅ Successfully uploaded 423 chunks from 3 PDFs!

Enter your choice: 2

💬 Enter your question: What were the causes of the Revolt of 1857?

🔍 Retrieving relevant content...
✓ Found relevant content from: history_ncert.pdf

🤖 Generating answer using llama3.2...

💡 Answer:
------------------------------------------------------------
The Revolt of 1857 had multiple causes:

1. **Political Causes**: The British policy of annexation through the 
   Doctrine of Lapse angered many Indian rulers...

2. **Economic Causes**: British policies destroyed traditional Indian 
   industries and imposed heavy taxes on farmers...

3. **Social and Religious Causes**: The British introduced reforms that 
   Indians perceived as interference in their customs...

4. **Military Causes**: Indian sepoys were dissatisfied with low pay, 
   discrimination, and the introduction of the Enfield rifle...
------------------------------------------------------------

📚 Sources: history_ncert.pdf
```

## Project Structure

```
rag/
├── main.py              # Main application
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── data/               # Your PDF files (create this)
│   ├── history_ncert.pdf
│   ├── polity_notes.pdf
│   └── ...
└── chroma_db/          # Vector database (auto-created)
```

## Configuration

You can customize the RAG system by modifying these parameters in `main.py`:

- **Chunk size**: `chunk_size=500` (number of words per chunk)
- **Chunk overlap**: `overlap=50` (words shared between chunks)
- **Number of results**: `n_results=5` (chunks retrieved per query)
- **Embedding model**: `all-MiniLM-L6-v2` (change in `__init__`)
- **LLM model**: `llama3.2` (change in `ask()` method)

## Troubleshooting

### Ollama Connection Error

```bash
# Make sure Ollama is running
ollama serve

# In another terminal, verify it's working
ollama list
```

### Model Not Found

```bash
# Pull the required model
ollama pull llama3.2
```

### No Text Extracted from PDF

- Ensure PDFs are not image-based (scanned documents need OCR)
- Try a different PDF parser if needed

### Slow Performance

- Use a smaller model: `llama3.2` instead of `llama3.1`
- Reduce `n_results` to retrieve fewer chunks
- Increase `chunk_size` to have fewer total chunks

## Advanced Usage

### Use Different Models

```python
# In the ask() method, change the model parameter
rag.ask("Your question here", model="mistral")
```

### Programmatic Usage

```python
from main import RAGSystem

# Initialize
rag = RAGSystem(data_folder="my_pdfs", db_folder="my_db")

# Upload documents
rag.upload_pdfs()

# Query
rag.ask("What is democracy?")
```

## Requirements

- Python 3.8+
- Ollama installed and running
- At least 4GB RAM for embeddings and LLM
- Disk space for PDFs and vector database

## Planning to design the frontend 
## License

MIT License - Feel free to use and modify!
