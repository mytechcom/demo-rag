import uuid
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Callable


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str,
                 embed_func: Callable[[str], List[float]]):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embed_func = embed_func

    def add_texts(self, texts: List[str], metadatas: List[dict],
                  ids: List[str] = None) -> None:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        # 计算所有文本的向量
        embeddings = [self.embed_func(text) for text in texts]
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.embed_func(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        # 整理返回结构
        docs = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                docs.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": results["distances"][0][i] if results.get("distances") else None,
                })
        return docs

    def delete_by_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})

    def get_sources(self) -> List[str]:
        # 获取所有不重复的 source
        all_meta = self.collection.get()["metadatas"]
        if not all_meta:
            return []
        return list({meta["source"] for meta in all_meta if "source" in meta})