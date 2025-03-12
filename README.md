# 🚀 Semantic Search Library

A flexible library for semantic search supporting multiple backends (cosine similarity, FAISS, ScaNN, Chroma DB). It allows you to ingest CSV data, format it with custom templates, use either pre-trained or custom models, and perform similarity searches. The library is usable as both a standard Python package and via a command-line interface (CLI).

## ✨ Features

- 📂 **Data Ingestion:** Import CSV files and define custom templates (e.g., "`{name} {family} has the age: {age}`").
- 🧠 **Multiple Similarity Backends:** Choose from native cosine similarity, FAISS, ScaNN (optional), or Chroma DB.
- 🔧 **Flexible Model Options:** Use pre-trained SentenceTransformer models or load your own custom model.
- 🔌 **Dual Interface:** Use as a library in your own projects or interact via a CLI.
- 🌐 **Web API Example:** A sample FastAPI server is provided (`main.py`).

## 📦 Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/your-username/semantic-search-lib.git
cd semantic-search-lib
pip install -e .
```

> 🟠 **Note:**
> - If you want to use the ScaNN backend, follow the instructions in `semantic_search/backends/scann_backend.py` (ScaNN may require manual installation).
> - If you want to use the ChromaDB backend, install it with:

```bash
pip install chromadb
```

## 📘 Usage as a Library

Import and use the library in your Python project:

```python
import pandas as pd
from semantic_search import SemanticSearch

# Initialize the search engine with your desired backend and model
search_engine = SemanticSearch(backend="faiss", model_name="all-MiniLM-L6-v2")

# Build the index from a CSV file using a custom template
search_engine.build_index_from_csv("data.csv", "{name} {family} has the age: {age}")

# Perform a search query
distances, indices = search_engine.query("John Doe has the age: 30", top_k=5)

# Load the original data to display results
data = pd.read_csv("data.csv")

# Show the top-k results
results = data.iloc[indices]
results['distance'] = distances
print(results)
```

## 🛠️ Usage via CLI

1. **Build the Index:**

```bash
python cli.py index \
  --csv data.csv \
  --template "{name} {family} has the age: {age}" \
  --backend faiss \
  --model_name all-MiniLM-L6-v2
```

2. **Perform a Search:**

```bash
python cli.py search --query "John Doe has the age: 30" --top_k 5
```

## 🚀 Running the FastAPI Server

Start the API server with:

```bash
uvicorn main:app --reload
```

### 🛜 Available endpoints:

- **POST `/index/`:** Build the index by providing JSON with keys `csv_path`, `template`, `backend`, `model` (optional), and `model_name`.
- **POST `/search/`:** Perform a search by providing JSON with `query` and `top_k`.

---

🔧 **Happy Searching!** 🚀 Let me know if you’d like me to refine this further or add anything! ✌️

