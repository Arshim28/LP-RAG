from typing import List, Optional, Dict, Any

from google import genai
from google.genai import types
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.llms.gemini import Gemini

from src.config import GOOGLE_API_KEY
from src.caching import Cache

class QueryProcessor:
	def __init__(
		self,
		index: VectorStoreIndex,
		llm: Optional[LLM] = None,
		cache: Optional[Cache] = None,
		api_key: str = GOOGLE_API_KEY
	):
		self.index = index
		self.api_key = api_key
		self.client = genai.Client(api_key=self.api_key)

		if llm is None:
			self.llm = Gemini(
				model_name="gemini-2.0-flash",
				api_key=self.api_key
			)

		else:
			self.llm = llm

		self.cache = cache

	def _retrieve_from_index(self, query: str, top_k: int = 10):
		retriever = self.index.as_retriever(
			similarity_top_k=top_k,
		)

		return retriever.retrieve(query)

	def process_query(self, query: str, top_k: int = 10):
		if self.cache:
			cached_results = self.cache.get_query_results(query)
			if cached_results:
				return cached_results

		results = self._retrieve_from_index(query, top_k)

		if self.cache and results:
			serializable_results = [
				{
					"text": node.text,
					"score": node.score if hasattr(node, "score") else 0.0,
					"metadata": node.metadata
				}
				for node in results
			]

			self.cache.cache_query_results(query, serializable_results)

		return results

	def generate_answer(self, query: str, top_k: int = 5):
		retriever = self.index.as_retriever(similarity_top=top_k)
		query_engine = self.index.as_query_engine(
			llm=self.llm,
			retriever=retriever,
		)

		response = query_engine.query(query)
		return response

