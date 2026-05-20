import os

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, \
    RecursiveCharacterTextSplitter



def get_splitters():
    headers = [
        ('#', 'h1'),
        ('##', 'h2'),
        ('###', 'h3'),
    ]

    markdown_splitter = (
        MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    )
    recursive_splitter = (
        RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    )

    return markdown_splitter, recursive_splitter

def chunk(markdown, source, markdown_splitter, recursive_splitter):
    header_docs = markdown_splitter.split_text(markdown)
    final_docs = recursive_splitter.split_documents(header_docs)

    processed = []
    for doc in final_docs:
        processed.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    'source': source,
                    **doc.metadata
                }
            )
        )

    return processed

def process_docs(raw_dir):
    all_docs = []

    markdown_splitter, recursive_splitter = get_splitters()

    for filename in os.listdir(raw_dir):
        if not filename.endswith('.md'):
            continue
        path = os.path.join(raw_dir, filename)
        with open(path, 'r', encoding='utf-8') as f:
            markdown = f.read()

        processed_docs = chunk(
            markdown=markdown,
            source=filename,
            markdown_splitter=markdown_splitter,
            recursive_splitter=recursive_splitter
        )

        all_docs.extend(processed_docs)

    return all_docs


if __name__=='__main__':
    from constants import RAW_DIR

    docs = process_docs(raw_dir=RAW_DIR)