from typing import List

from google import genai
from google.genai import types

from src.config import GOOGLE_API_KEY, EMBEDDING_MODEL

class EmbeddingGenerator:
	def __init__(self, api_key=GOOGLE_API_KEY, model_name=EMBEDDING_MODEL):
		self.model = model_name
		self.api_key = api_key

		self.client = genai.Client(api_key=self.api_key)
		self.config = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")

	def embed_texts(self, texts: List[str], batch_size=100) -> List[List[float]]:
		all_embeddings = []

		for i in range(0, len(texts), batch_size):
			batch = texts[i:i+batch_size]
			batch_embedding = self.client.models.embed_content(
				model=self.model,
				contents=batch,
				config=self.config,
			)
			all_embeddings.extend(batch_embedding.embeddings)

		return all_embeddings

	def embed_query(self, query: str) -> List[float]:
		return self.client.models.embed_content(model=self.model, contents=query, config=self.config).embeddings[0]