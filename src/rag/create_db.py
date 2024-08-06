import json
import os
import shutil

import numpy as np
import pandas as pd
import requests
from langchain.docstore.document import Document
from langchain.indexes import SQLRecordManager, index
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI


class Embedder:
    def __init__(self, model="second-state/All-MiniLM-L6-v2-Embedding-GGUF"):
        self.model = model
        self.client = OpenAI(base_url="http://localhost:5000/v1", api_key="lm-studio")

    def embed_documents(self, input):
        return [d.embedding for d in self.client.embeddings.create(input=input, model=self.model).data]

    def embed_query(self, query: str):
        return self.client.embeddings.create(input=query, model=self.model).data[0].embedding


DATA_PATH = "data/arxiv_metadata.parquet.gzip"
CAT_PATH = "data/categories.json"


def load_data():
    data = pd.read_parquet(DATA_PATH)
    categories = json.load(open(CAT_PATH, "r", encoding="utf-8"))
    return data, categories


CHROMA_PATH = "."
COLLECTION_NAME = "master"


def create_vectorstore():
    embedder = Embedder()
    vectorstore = Chroma(collection_name=COLLECTION_NAME, embedding_function=embedder, persist_directory=CHROMA_PATH)
    return vectorstore


def get_vectorstore():
    assert os.path.exists(CHROMA_PATH), f"Chroma vectorstore path {CHROMA_PATH} does not exist"
    return Chroma(collection_name=COLLECTION_NAME, persist_directory=CHROMA_PATH, create_collection_if_not_exists=False)


def add(data, recorder, vectorstore):

    return index(data, record_manager=recorder, vector_store=vectorstore, cleanup="full", source_id_key="source")


def delete_vectorstore():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def preprocess_data(df):
    docs = []
    for i, row in df.iterrows():
        doc = Document(page_content=row["abstract"], metadata={"source": row["id"]})
        docs.append(doc)

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=512, chunk_overlap=64, add_start_index=True # ~ 2.35x
    # )
    # docs = text_splitter.split_documents(docs)
    return docs


if __name__ == "__main__":
    data, categories = load_data()
    documents = preprocess_data(data)
    print(f"Data loaded: {len(documents)} documents.")
    # delete_vectorstore()
    vectorstore = create_vectorstore()
    print(f"Currently {len(vectorstore.get()['ids'])} documents in vectorstore.")
    recorder = SQLRecordManager(namespace=f"arxiv/{COLLECTION_NAME}", db_url="sqlite:///chroma.sql")
    recorder.create_schema()
    print("Adding documents to vectorstore...")
    res = add(documents, recorder, vectorstore)

    # vectorstore.get(): dict_keys(['ids', 'embeddings', 'metadatas', 'documents', 'uris', 'data'])
    print(f"Added {len(vectorstore.get()['ids'])} documents to vectorstore.")
