import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline
from src.config import REPORTS_DIR, INDEXES_DIR

def truncate_text(text, max_length=300):
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def run_demo():
    pipeline = RAGPipeline(use_cache=True)
    
    index_path = INDEXES_DIR / "financial_reports.faiss"
    
    if index_path.exists():
        print("Loading existing index...")
        index = pipeline.load_existing_index()
        if not index:
            print("Failed to load index. Creating a new one...")
            create_new_index(pipeline)
    else:
        print("No existing index found. Creating a new one...")
        create_new_index(pipeline)
    
    while True:
        user_query = input("\nEnter your query (or 'exit' to quit): ")
        
        if user_query.lower() in ['exit', 'quit', 'q']:
            break
        
        try:
            print("\nSearching for relevant information...")
            
            results = pipeline.query(user_query, top_k=10, rerank=True, generate_answer=False)
            
            if not results:
                print("No relevant information found.")
                continue
                
            print("\n=== Top Results ===")
            for i, result in enumerate(results[:5]):
                text = result.text if hasattr(result, 'text') else result.node.text
                metadata = result.metadata if hasattr(result, 'metadata') else result.node.metadata
                
                print(f"\n--- Result {i+1} ---")
                print(f"Source: {metadata.get('report_name', 'Unknown')}")
                print(truncate_text(text))
            
            generate = input("\nGenerate an answer based on these results? (y/n): ")
            if generate.lower() in ['y', 'yes']:
                print("\nGenerating answer...")
                answer = pipeline.query(user_query, top_k=5, rerank=True, generate_answer=True)
                print("\n=== Generated Answer ===")
                print(answer["answer"])
                
        except Exception as e:
            print(f"An error occurred: {e}")
        
        print("\n" + "=" * 50)

def create_new_index(pipeline):
    reports = list(REPORTS_DIR.glob("*.pdf"))
    
    if not reports:
        print(f"No PDF reports found in {REPORTS_DIR}. Please add some reports first.")
        sys.exit(1)
    
    print(f"Found {len(reports)} reports. Ingesting and indexing...")
    
    try:
        pipeline.ingest_and_index(reports)
        print("Indexing complete!")
    except Exception as e:
        print(f"Error during indexing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_demo()