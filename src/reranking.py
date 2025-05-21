from typing import List, Tuple, Any

from google import genai
from google.genai import types

from src.config import GOOGLE_API_KEY

class Reranker:
	def __init__(self, api_key=GOOGLE_API_KEY, model_name="gemini-2.5-pro-exp-03-25"):
		self.api_key = api_key
		self.model_name = model_name
		self.client = genai.Client(api_key=self.api_key)

	def _score_chunk(self, chunk: Any, query: str) -> float:
		prompt = f"""
		Rate the relevance of this excerpt to the query on a scle of 0-10.
		Only respond with a number between 0 and 10, nothing else.

		Query: {query}

		Excerpt: {chunk.text if hasattr(chunk, 'text') else chunk.node.text}

		Relevance score (0-10):
		"""

		try:
			response = self.client.models.generate_text(
				model=self.model_name,
				prompt=prompt,
				temperature=0.1,
				max_output_tokens=10,
			)
			score = float(response.text.strip())
			if score < 0 or score > 10:
				raise ValueError(f"Score out of range: {score}")

			return score
		except Exception as e:
			print(f"Error scoring chunk: {e}")
			return 0.0

	def rerank(self, chunks: List[Any], query: str) -> List[Tuple[Any, float]]:
		scored_chunks = []

		for chunk in chunks:
			score = self._score_chunk(chunk, query)
			scored_chunks.append((chunk, score))

		return sorted(scored_chunks, key=lambda x: x[1], reverse=True)