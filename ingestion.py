import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone


pinecone.init(api_key=,environment=os.environ["PINECONE_ENVIRONMENT_REGION"])



def ingest_docs() -> None:
    loader = ReadTheDocsLoader(path="langchain-docs",encoding='UTF-8')
    raw_documents = loader.load()
    print(f"loaded{len(raw_documents)}documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100,
                                                   separators=["\n\n", "\n", " ", ""])
    documents = text_splitter.split_documents(documents=raw_documents)

    print(f'Splitted into {len(documents)} chunks')


    for doc in documents:
        old_path=doc.metadata['source']
        new_url= old_path.replace("langchain-docs","https:/")
        doc.metadata.update({"source":new_url})

    print(f"Going to insert{len(documents)} in Pinecone")
    embeddings=OpenAIEmbeddings()
    Pinecone.from_documents(documents[3990:], embeddings, index_name="langchain-doc-index")
    print("Added to Pinecone")

if __name__ == '__main__':
    ingest_docs()
