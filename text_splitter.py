import re
import numpy as np
from typing import List, Dict, Callable


class SimpleSplitter:
    """固定大小切分"""
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> List[Dict]:
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end]
            chunks.append({"text": chunk_text, "metadata": {}})
            start += self.chunk_size - self.overlap
        return chunks


class ParentChildSplitter:
    """父子切分：父块较大，子块较小；子块入库并关联父块"""
    def __init__(self, parent_size: int = 1000, parent_overlap: int = 200,
                 child_size: int = 300, child_overlap: int = 50):
        self.parent_splitter = SimpleSplitter(parent_size, parent_overlap)
        self.child_size = child_size
        self.child_overlap = child_overlap

    def split(self, text: str) -> List[Dict]:
        parent_chunks = self.parent_splitter.split(text)
        all_children = []
        for p_idx, parent in enumerate(parent_chunks):
            parent_text = parent["text"]
            child_splitter = SimpleSplitter(self.child_size, self.child_overlap)
            children = child_splitter.split(parent_text)
            for child in children:
                all_children.append({
                    "text": child["text"],
                    "metadata": {
                        "parent_id": f"p{p_idx}",
                        "parent_text": parent_text
                    }
                })
        return all_children


class SemanticSplitter:
    """语义切分：基于相邻句子 embedding 的余弦相似度切分"""
    def __init__(self, embed_func: Callable[[str], List[float]],
                 similarity_threshold: float = 0.7,
                 max_chunk_size: int = 800):
        self.embed_func = embed_func
        self.threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size

    def _split_sentences(self, text: str) -> List[str]:
        # 简单按中英文标点切句
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def split(self, text: str) -> List[Dict]:
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [{"text": text, "metadata": {}}] if text else []

        # 获取每个句子的向量
        embeddings = [self.embed_func(sent) for sent in sentences]
        # 计算相邻句子余弦相似度
        similarities = []
        for i in range(len(embeddings) - 1):
            a = np.array(embeddings[i])
            b = np.array(embeddings[i+1])
            sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
            similarities.append(sim)

        # 确定切分点（相似度低于阈值的位置）
        split_points = [i for i, sim in enumerate(similarities) if sim < self.threshold]

        # 根据切分点合并句子，并限制最大长度
        chunks = []
        start = 0
        for point in split_points:
            end = point + 1
            segment = " ".join(sentences[start:end])
            # 如果当前段超过最大长度，进一步按长度切分
            if len(segment) > self.max_chunk_size:
                short_chunks = self._force_split(segment)
                chunks.extend(short_chunks)
            else:
                if segment.strip():
                    chunks.append({"text": segment, "metadata": {}})
            start = end
        # 最后一段
        if start < len(sentences):
            segment = " ".join(sentences[start:])
            if len(segment) > self.max_chunk_size:
                short_chunks = self._force_split(segment)
                chunks.extend(short_chunks)
            else:
                if segment.strip():
                    chunks.append({"text": segment, "metadata": {}})
        return chunks

    def _force_split(self, text: str) -> List[Dict]:
        # 简单固定长度切分，保持元数据结构一致
        splitter = SimpleSplitter(self.max_chunk_size, overlap=50)
        return splitter.split(text)