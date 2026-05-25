from langchain_core.output_parsers import StrOutputParser

from rag.generation import Generator
from rag.prompt import prompt_template
from rag.reranker import Reranker
from rag.retrieval import Retriever
from rag.validator import Validator


class RAGPipeline:
    def __init__(
        self, 
        idx_path: str, 
        metadata_path: str, 
        embedding_model: str,
        reranker_model: str,
        generator_model: str,
    ):
        self.retriever = Retriever(
            idx_path=idx_path,
            metadata_path=metadata_path,
            embedding_model=embedding_model,
        )
        self.reranker = Reranker(reranker_model=reranker_model)
        self.validator = Validator()
        self.generator = Generator(model_name=generator_model)
        
        self.chain = (
            prompt_template
            | self.generator.llm
            | StrOutputParser()
        )

    def answer(self, question: str) -> str:
        docs = self.retriever.invoke(question)

        ranked = self.reranker.rerank(query, docs)

        if not self.validator.validate(ranked):
            return "Fallback: I could not find the answer in the documentation"

        context_docs = [
            doc
            for doc, _ in ranked[:3]
        ]

        context = "\n\n".join(doc.page_content for doc in context_docs)
        return self.chain.invoke({
            'question': question,
            'context': context,
        })
    
if __name__=='__main__':
    from constants import EMBEDDING_MODEL, GENERATOR_MODEL, \
        IDX_PATH, METADATA_PATH
    
    query = 'What is a Python generator?'
    rag = RAGPipeline(
        idx_path=IDX_PATH,
        metadata_path=METADATA_PATH,
        embedding_model=EMBEDDING_MODEL,
        generator_model=GENERATOR_MODEL,
    )

    result = rag.answer(query)
    print(result)
