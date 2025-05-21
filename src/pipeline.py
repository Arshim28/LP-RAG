import json
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any

from src.document_ingestion import DocumentIngestion
from src.embedding import EmbeddingGenerator
from src.indexing import Indexer
from src.caching import Cache
from src.query_processing import QueryProcessor
from src.reranking import Reranker
from src.config import REPORTS_DIR

class RAGPipeline:
	def __init__(self, use_cache: bool = True):
		self.embedding_generator = EmbeddingGenerator()
		self.document_ingestion = DocumentIngestion()
		self.indexer = Indexer(self.embedding_generator)
		self.cache = Cache() if use_cache else None
		self.reranker = Reranker()
		self.query_processor = None

	def ingest_and_index(
		self,
		report_path: Union[str, Path, List[str], List[Path]],
		index_name: str = "financial_reports"
	) -> Any:

		if isinstance(report_path, (str, Path)):
			report_path = [report_path]

		parsed_paths = []
		for path in report_path:
			path = Path(path)
			if path.exists() and path.suffix.lower() == ".pdf":
				parsed_path = self.document_ingestion.parse_pdf(str(path))
				parsed_paths.append(parsed_path)

		if not parsed_paths:
			raise ValueError("No valid PDF reports found at the provided path(s)")

		metadata_list = []
		for i, path in enumerate(report_path):
			path = Path(path)
			metadata_list.append({
				"report_name": path.name,
				"report_id": f"report_{i}",
				"file_path": str(path),
			})

		documents = self.indexer.create_documents(parsed_paths, metadata_list)
		index = self.indexer.build_index(documents, index_name)

		self.query_processor = QueryProcessor(index, cache=self.cache)

		return index

	def load_existing_index(self, index_name: str = "financial_reports") -> Optional[Any]:
		index = self.indexer.load_index(index_name)

		if index:
			self.query_processor = QueryProcessor(index, cache=self.cache)
			return index

		return None

	def query(
		self,
		user_query: str,
		top_k: int = 10,
		rerank: bool = True,
		generate_answer: bool = False
	) -> Any:
		
		if not self.query_processor:
			raise ValueError("No indexes loaded. Please ingest and index the docs first or load an existing index")

		results = self.query_processor.process_query(user_query, top_k=top_k)

		if rerank and results:
			ranked_results = self.reranker.rerank(results, user_query)
			top_chunks = [chunk for chunk, _ in ranked_results]

		else:
			top_chunks = results

		if generate_anwer:
			answer = self.query_processor.generate_answer(user_query)
			return {
				"answer": answer,
				"sources": top_chunks[:5] if top_chunks else []
			}

		return top_chunks

