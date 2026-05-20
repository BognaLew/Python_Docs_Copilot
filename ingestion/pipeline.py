from build_index import build_faiss
from chunk import process_docs
from crawler import crawl


def ingestion_pipeline(base_url, raw_dir):
    crawl(base_url=base_url, raw_dir=raw_dir)
    docs = process_docs(raw_dir=raw_dir)
    build_faiss(docs=docs)


if __name__=='__main__':
    from constants import BASE_URL, RAW_DIR

    ingestion_pipeline(base_url=BASE_URL, raw_dir=RAW_DIR)