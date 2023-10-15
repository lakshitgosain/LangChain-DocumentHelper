import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import  Pinecone
import pinecone
#from const import INDEX_NAME
INDEX_NAME="langchain-doc-index"

pinecone.init(
    api_key=,#os.environ["PINECONE_API_KEY"],
    environment='gcp-starter',#os.environ["PINECONE_ENVIRONMENT_REGION"],
)

def run_llm(query:str, chat_history) :

    embeddings=OpenAIEmbeddings()
    docsearch= Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)

    chat=ChatOpenAI(verbose=True, temperature=0)
    #qa=RetrievalQA.from_chain_type(llm=chat, chain_type='stuff', return_source_documents= True ,retriever=docsearch.as_retriever())

    qa=ConversationalRetrievalChain.from_llm(llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True)


    return qa({'question':query, "chat_history": chat_history})


if __name__=='__main__':
    print(run_llm(query='What is langchain'))






