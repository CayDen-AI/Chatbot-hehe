from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import hub
from langchain.chains.retrieval_qa.base import RetrievalQA
from base_model import get_model
from utils import extract_answer


def read_db(docs):
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    db = FAISS.load_local(docs, embedding_model, allow_dangerous_deserialization=True)
    return db


def create_prompt():
    return hub.pull("rlm/rag-prompt-llama")


def create_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 3}, max_tokens_limit=512),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}
    )
    return llm_chain


# if __name__ == '__main__':
#     FAISS_PATH = 'faiss'
#     MODEL_NAME = 'vilm/vinallama-2.7b-chat'
#     db = read_db(FAISS_PATH)
#     llm = get_model(MODEL_NAME)
#     prompt = create_prompt()
#     llm_chain = create_chain(prompt, llm, db)
#
#     question = 'chatbot là gì?'
#     response = llm_chain.invoke({'query': question})
#     answer = extract_answer(response['result'])
#     print(answer)
