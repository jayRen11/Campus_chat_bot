import os
import chromadb
from chromadb.utils import embedding_functions

# 确保上一行没有乱入的 import 语句！

class DBManager:
    def __init__(self, db_path="./campus_knowledge_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.life_db = self.client.get_or_create_collection("regulations", embedding_function=self.embedding_fn)
        self.study_db = self.client.get_or_create_collection("study_materials", embedding_function=self.embedding_fn)

    def get_collection(self, mode):
        return self.life_db if mode == "生活助手" else self.study_db

    def search(self, query, mode, top_k=5):
        target_db = self.get_collection(mode)
        if target_db.count() == 0:
            return ""
        results = target_db.query(query_texts=[query], n_results=top_k)
        return "\n".join(results['documents'][0])

    def ingest(self, chunks, mode, file_name):
        target_db = self.get_collection(mode)
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
        target_db.add(documents=chunks, ids=ids)
        return len(chunks)

    def clear(self, mode):
        collection_name = "regulations" if mode == "生活助手" else "study_materials"
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        if mode == "生活助手":
            self.life_db = self.client.get_or_create_collection("regulations", embedding_function=self.embedding_fn)
        else:
            self.study_db = self.client.get_or_create_collection("study_materials", embedding_function=self.embedding_fn)

    def count(self, mode):
        return self.get_collection(mode).count()
