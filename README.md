# Financial Reports RAG Pipeline

A Python-based Retrieval-Augmented Generation (RAG) pipeline for extracting insights from annual reports and financial documents.

## Features

- **Document Ingestion**: Parse complex PDFs into structured text using LlamaParse
- **Embedding & Indexing**: Generate high-quality embeddings with Google Gemini and index with FAISS
- **Caching**: Redis-based caching for faster retrieval and reduced API costs
- **Query Processing**: Process natural language queries with Google Gemini
- **Reranking**: Refine search results with LLM-based relevance scoring
- **Hybrid Answers**: Generate complete answers or retrieve relevant passages

## Requirements

- Python 3.13 
- Redis server (for caching)
- LlamaParse API key
- Google Gemini API key

## Getting Started

1. Clone this repository
2. Install dependencies:
   ```
   uv sync
   ```
3. Update the created `.env` file with your API keys
4. Place your PDF reports in the `data/reports/` directory
5. Run the demo:
   ```
   uv run -m scripts/demo.py
   ```

## Usage Example

```python
from src.pipeline import RAGPipeline

# Initialize the pipeline
pipeline = RAGPipeline(use_cache=True)

# Option 1: Ingest and index reports
pipeline.ingest_and_index("path/to/annual_report_2024.pdf")

# Option 2: Load an existing index
pipeline.load_existing_index()

# Retrieve relevant passages
results = pipeline.query(
    "What were the major risk factors in FY 2024?",
    top_k=5,
    rerank=True
)

# Get a generated answer
response = pipeline.query(
    "Summarize the company's ESG initiatives",
    generate_answer=True
)
print(response["answer"])
```

## Architecture

This RAG pipeline follows a six-stage architecture:

1. **Document Ingestion**: Convert PDFs to structured text using LlamaParse
2. **Embedding Generation**: Create vector representations using Google Gemini
3. **Indexing**: Store vectors in FAISS for efficient similarity search
4. **Caching**: Cache results in Redis to improve performance
5. **Query Processing**: Process natural language queries and retrieve relevant passages
6. **Reranking & Generation**: Refine results and generate complete answers

## Customization

- Adjust chunking strategies in `document_ingestion.py`
- Modify embedding parameters in `embedding.py`
- Configure caching TTLs in `caching.py`
- Tune retrieval parameters in `query_processing.py`
