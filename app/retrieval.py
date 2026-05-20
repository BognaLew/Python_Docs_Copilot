import pickle

import faiss
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, idx_path, metadata_path, embedding_model):
        self.idx = faiss.read_index(idx_path)
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.model = SentenceTransformer(embedding_model)
    
    def search(self, query, k=5):
        v = self.model.encode([query])
        _, indices = self.idx.search(v, k)

        return [self.metadata[idx] for idx in indices[0]]
        
if __name__=='__main__':
    from constants import EMBEDDING_MODEL, IDX_PATH, METADATA_PATH


    retriever = Retriever(
        idx_path=IDX_PATH,
        metadata_path=METADATA_PATH,
        embedding_model=EMBEDDING_MODEL,
    )

    query = 'What is a Python generator?'
    result = retriever.search(query=query)
    print(result)
