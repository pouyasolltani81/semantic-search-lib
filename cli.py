import argparse
import sys
from semantic_search.search import SemanticSearch

# Global search engine instance for the CLI session
search_engine = None

def build_index(args):
    global search_engine
    try:
        search_engine = SemanticSearch(backend=args.backend, model_path=args.model, model_name=args.model_name)
        print('works till here')
        search_engine.build_index_from_csv(args.csv, args.template)
        print(f"Index built successfully using backend: {args.backend}")
    except Exception as e:
        print(f"Error building index: {e}")
        sys.exit(1)

def perform_search(args):
    global search_engine
    if search_engine is None:
        print("Error: Index not built. Use the 'index' command first.")
        sys.exit(1)
    try:
        distances, indices = search_engine.query(args.query, args.top_k)
        print("\nSearch Results:")
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            print(f"{i+1}. Result index {idx} - Similarity: {dist:.4f}")
    except Exception as e:
        print(f"Error during search: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Command: index
    index_parser = subparsers.add_parser("index", help="Build the index from a CSV file")
    index_parser.add_argument("--csv", required=True, help="Path to CSV file")
    index_parser.add_argument("--template", required=True,
                              help="Template for each row (e.g., '{name} {family} has the age: {age}')")
    index_parser.add_argument("--backend", choices=["cosine", "faiss", "scann", "chromadb"],
                              default="cosine", help="Similarity backend to use")
    index_parser.add_argument("--model", help="Path to a custom model (optional)")
    index_parser.add_argument("--model_name", default="all-MiniLM-L6-v2",
                              help="Pre-trained model name (if not using a custom model)")

    # Command: search
    search_parser = subparsers.add_parser("search", help="Search the built index")
    search_parser.add_argument("--query", required=True, help="Search query text")
    search_parser.add_argument("--top_k", type=int, default=5, help="Number of top results to return")

    args = parser.parse_args()

    if args.command == "index":
        build_index(args)
    elif args.command == "search":
        perform_search(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
