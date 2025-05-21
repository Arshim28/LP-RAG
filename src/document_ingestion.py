from pathlib import Path

from llama_parse import LlamaParse

from src.config import LLAMA_API_KEY, PARSED_DIR

class DocumentIngestion:
	def __init__(self, api_key=LLAMA_API_KEY, result_type="markdown", num_workers=4):
		self.parser = LlamaParse(api_key=api_key, result_type=result_type, num_workers=num_workers)

	def parse_pdf(self, pdf_path: str, output_filename: str = None):
		pdf_path = Path(pdf_path)

		if output_filename is None:
			output_filename = f"{pdf_path.stem}.md"

		output_path = PARSED_DIR / output_filename
		result = self.parser.parse(str(pdf_path), output_filename=str(output_path))

		return output_path

	def parse_multiple_pdfs(self, pdf_paths):
		return [self.parse_pdf(path) for path in pdf_paths] # -> make this parallel in future

