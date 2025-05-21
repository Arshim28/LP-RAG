import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = int(os.getenv("REDIS_DB", 0))

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = DATA_DIR / "reports"
PARSED_DIR = DATA_DIR / "parsed"
INDEXES_DIR = DATA_DIR / "indexes"

EMBEDDING_MODEl = "text-embedding-004"

for dir_path in [DATA_DIR, REPORTS_DIR, PARSED_DIR, INDEXES_DIR]:
	dir_path.mkdir(exist_ok=True, parents=True)

