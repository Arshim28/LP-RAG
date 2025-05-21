import json
from typing import Any, List, Dict, Optional

import redis

from src.config import REDIS_HOST, REDIS_PORT, REDIS_DB

class Cache:
	def __init__(self, host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB):
		self.redis = redis.Redis(host=host, port=port, db=db)

	def get_query_results(self, query: str) -> Optional[List[Dict]]:
		key = f"query:{query}"
		result = self.redis.get(key)

		if result:
			return json.loads(result)
		return None

	def cache_query_results(self, query: str, results: List[Dict], ttl: int = 3600):
		key = f"query:{query}"
		self.redis.setex(key, ttl, json.dumps(results))

	def get_embedding(self, text: str) -> Optional[List[float]]:
		key = f"embedding:{text}"
		result = self.redis.get(key)

		if result:
			return json.loads(result)
		return None

	def cache_embedding(self, text: str, embedding:	List[float], ttl: int = 86400):
		key = f"embedding:{text}"
		self.redis.setex(key, ttl, json.dumps(embedding))

	def get_document(self, doc_id: str) -> Optional[Dict]:
		key = f"document:{doc_id}"
		result = self.redis.get(key)

		if result:
			return json.loads(result)
		return None

	def cache_document(self, doc_id: str, document: Dict, ttl: int = 86400 * 7):
		key = f"document:{doc_id}"
		self.redis.setex(key, ttl, json.dumps(document))

	def clear_cache(self, pattern: str = "*"):
		for key in self.redis.scan_iter(f"{pattern}"):
			self.redis.delete(key)


