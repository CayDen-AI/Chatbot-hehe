import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.schema import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


DATA_PATH = 'data/fithou'
FAISS_PATH = 'faiss'


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    docs = loader.load()

    return docs


def split_text(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', ' ', ''],
        chunk_size=300,
        chunk_overlap=10,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks.")

    return chunks


def save_to_faiss(chunks: list[Document]):
    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)

    # embedding_model = GPT4AllEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(FAISS_PATH)
    print(f"Saved {len(chunks)} chunks to {FAISS_PATH}.")


def main():
    docs = load_documents()
    chunks = split_text(docs)
    save_to_faiss(chunks)


if __name__ == "__main__":
    main()
