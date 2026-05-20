import pickle

import faiss
import numpy as np
import torch


def build_faiss(docs, embedding_model, idx_path, metadata_path, batch_size=64):
    texts = [doc.page_content for doc in docs]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = embedding_model.to(device)

    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device,
    )
    
    idx = faiss.IndexFlatL2(embeddings.shape[1])

    idx.add(np.array(embeddings).astype('float32'))

    metadata = []
    for doc in docs:
        metadata.append({
            'text': doc.page_content,
            'metadata': doc.metadata,
        })

    faiss.write_index(
        idx,
        idx_path,
    )

    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)


if __name__=='__main__':
    from sentence_transformers import SentenceTransformer

    from constants import IDX_PATH, METADATA_PATH, RAW_DIR
    from ingestion.chunk import process_docs


    docs = process_docs(raw_dir=RAW_DIR)

    embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    build_faiss(
        docs=docs,
        embedding_model=embedding_model,
        idx_path=IDX_PATH,
        metadata_path=METADATA_PATH,
    )