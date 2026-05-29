from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, reranker_model: str):
        self.model = CrossEncoder(reranker_model)

    def rerank(self, query: str, docs: List[Document]):
        pairs = [
            [query, doc.page_content]
            for doc in docs
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked
    