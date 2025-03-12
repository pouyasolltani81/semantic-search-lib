from semantic_search.embeddings import Embedder

try:
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    sentences = ["John Doe", "Jane Smith"]
    embeddings = embedder.encode(sentences)
    print("Embeddings shape:", embeddings.shape)  # Expect something like (2, 384)
except Exception as e:
    print("Error in embedding generation:", e)
