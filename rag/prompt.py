from langchain_core.prompts import PromptTemplate


TEMPLATE = """
You are an expert Python assistant.

Answer the question using ONLY provided context.

If the context does not explicitly contain the answer or there is 
no context provided, do not guess and respond ONLY with:
'I could not find the answer in the documentation'

Question:
{question}

Context:
{context}

Answer:
"""

prompt_template = PromptTemplate(
    template=TEMPLATE,
    input_variables=['question', 'context'],
)
