from pathlib import Path
from typing import List, Dict, Optional, Union

from llama_index.core import (
	VectorStoreIndex,
	Document,
	Settings,
	StorageContext
)
from llama_index.vector_stores.faiss import FaissVectorStore

from src.config import INDEXES_DIR

class Indexer:
	def __init__(self, embed_model):
		Settings.embed_model = embed_model
		self.embed_model = embed_model

	def create_documents(
		self,
		parsed_paths: Union[str, List[str]],
		metadata: Optional[Union[Dict, List[Dict]]] = None
	) -> List[Document]:
		documents = []

		if isinstance(parsed_paths, (str, Path)):
			parsed_paths = [parsed_paths]

		if metadata is None:
			metadata = [{}] * len(parsed_paths)

		elif isinstance(metadata, dict):
			metadata = [metadata]

		for path, meta in zip(parsed_paths, metadata):
			path = Path(path)
			if path.exists():
				with open(path, 'r', encoding='utf-8') as f:
					content = f.read()

				meta_with_source = meta.copy()
				meta_with_source['source'] = str(path)
				documents.append(Document(text=content, metadata=meta_with_source))

		return documents

	def build_index(self, documents: List[Document], index_name: str = "financial_reports"):
		vector_store = FaissVectorStore()
		storage_context = StorageContext.from_defaults(vector_store=vector_store)

		index = VectorStoreIndex.from_documents(
			documents,
			storage_context=storage_context
		)

		index_path = INDEXES_DIR / f"{index_name}.faiss"
		index.storage_context.persist(persist_dir=str(index_path))

		return index

	def load_index(self, index_name: str = "financial_reports"):
		index_path = INDEXES_DIR / f"{index_name}.faiss"

		if index_path.exists():
			storage_context = StorageContext.from_defaults(
				vector_store=FaissVectorStore(),
				persist_dir=str(index_path)
			)
			return VectoreStoreIndex.from_storage_context(storage_context)

		return None