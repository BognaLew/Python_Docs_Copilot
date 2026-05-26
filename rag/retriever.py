import pickle
from typing import Any, List

import faiss
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr
from sentence_transformers import SentenceTransformer


class Retriever(BaseRetriever):
    _idx: faiss.Index = PrivateAttr()
    _metadata: list[dict[str, Any]] = PrivateAttr()
    _model: SentenceTransformer = PrivateAttr()
    k: int = 5

    def __init__(
        self, 
        idx_path: str, 
        metadata_path: str, 
        embedding_model: str,
        k: int = 5,
    ):
        super().__init__()

        self._idx = faiss.read_index(idx_path)
        with open(metadata_path, 'rb') as f:
            self._metadata = pickle.load(f)
        
        self._model = SentenceTransformer(embedding_model)
        self.k = k
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        v = self._model.encode([query])
        scores, indices = self._idx.search(v, self.k)

        documents = []

        if sum(scores[0][:self.k]) / self.k > 1.2:
            return documents

        for i in indices[0]:
            chunk = self._metadata[i]
            documents.append(
                Document(
                    page_content=chunk['text'],
                    metadata=chunk['metadata'],
                )
            )

        return documents
