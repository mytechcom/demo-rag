import os
from document_loader import load_document
from text_splitter import SimpleSplitter, ParentChildSplitter, SemanticSplitter
from vector_store import VectorStore
import config
from typing import List, Dict


class RAGEngine:
    def __init__(self, vector_store: VectorStore, chat_func, embed_func):
        self.vector_store = vector_store
        self.chat_func = chat_func
        self.embed_func = embed_func

    def index_file(self, file_path: str, splitter_type: str = "simple") -> None:
        # 加载文档
        text = load_document(file_path)
        source = os.path.basename(file_path)

        # 选择切分器
        if splitter_type == "simple":
            splitter = SimpleSplitter(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        elif splitter_type == "parent_child":
            splitter = ParentChildSplitter(
                config.PARENT_CHUNK_SIZE, config.PARENT_CHUNK_OVERLAP,
                config.CHILD_CHUNK_SIZE, config.CHILD_CHUNK_OVERLAP,
            )
        elif splitter_type == "semantic":
            splitter = SemanticSplitter(
                self.embed_func,
                config.SEMANTIC_SIMILARITY_THRESHOLD,
                config.MAX_SEMANTIC_CHUNK_SIZE,
            )
        else:
            raise ValueError(f"Unknown splitter type: {splitter_type}")

        chunks: List[Dict] = splitter.split(text)

        # 添加 source 信息到 metadata
        texts = []
        metadatas = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            meta["source"] = source
            texts.append(chunk["text"])
            metadatas.append(meta)

        # 插入向量库（如果之前有同名文件，先删除再插入）
        self.vector_store.delete_by_source(source)
        self.vector_store.add_texts(texts, metadatas)

    def query(self, question: str, top_k: int = 4) -> Dict:
        # 检索
        docs = self.vector_store.search(question, top_k=top_k)

        # 构建上下文：优先使用父块文本（父子切分时），否则用 chunk 文本
        context_parts = []
        source_display = []
        for doc in docs:
            parent = doc["metadata"].get("parent_text")
            display_text = parent if parent else doc["text"]
            context_parts.append(display_text)
            source_display.append({
                "source": doc["metadata"].get("source", "unknown"),
                "text": display_text,
                "score": doc.get("score"),
            })

        context = "\n\n---\n\n".join(context_parts)

        # 构造提示词
        prompt = f"""You are a helpful assistant. Use the following context to answer the question accurately. 
If you don't know the answer, just say you don't know.

Context:
{context}

Question: {question}
Answer:"""

        messages = [{"role": "user", "content": prompt}]
        answer = self.chat_func(messages)

        return {
            "answer": answer,
            "sources": source_display,
        }