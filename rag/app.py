import os

from fastapi import FastAPI
from pydantic import BaseModel

from rag_pipeline import RAGPipeline


app = FastAPI()

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
GENERATOR_MODEL = os.getenv('GENERATOR_MODEL')
IDX_PATH = os.getenv('IDX_PATH')
METADATA_PATH = os.getenv('METADATA_PATHH')
RERANKER_MODEL = os.getenv('IRERANKER_MODEL')

rag = RAGPipeline(
    idx_path=IDX_PATH,
    metadata_path=METADATA_PATH,
    embedding_model=EMBEDDING_MODEL,
    reranker_model=RERANKER_MODEL,
    generator_model=GENERATOR_MODEL,
)

def run_rag(query: str):
    result = rag.answer(query)

    return {
        'answer': result['answer'],
        'context': result['context'],
    }


class QueryRequest(BaseModel):
    question: str


@app.post('/query')
async def query_rag(request: QueryRequest):

    result = run_rag(request.question)

    return result