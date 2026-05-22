from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser

from app.generation import Generator
from app.prompt import prompt_template
from app.retrieval import Retriever

class RAGPipeline:
    def __init__(
        self, 
        idx_path: str, 
        metadata_path: str, 
        embedding_model: str,
        generator_model: str,
    ):
        self.retriever = Retriever(
            idx_path=idx_path,
            metadata_path=metadata_path,
            embedding_model=embedding_model,
        )

        self.generator = Generator(model_name=generator_model)
        
        self.chain = (
            prompt_template
            | self.generator.llm
            | StrOutputParser()
        )

    def answer(self, question: str) -> str:
        docs = self.retriever.invoke(question)

        if not docs:
            return "I could not find the answer in the documentation"

        context = "\n\n".join(doc.page_content for doc in docs)
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
