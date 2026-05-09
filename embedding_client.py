"""
本地嵌入模型客户端，基于 sentence-transformers 加载模型
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from config import LOCAL_EMBED_MODEL_PATH

# 全局单例，避免重复加载模型
_embed_model = None

def _get_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(LOCAL_EMBED_MODEL_PATH)
    return _embed_model

def get_embedding(text: str) -> list[float]:
    """
    返回文本的向量表示（list of float）
    """
    model = _get_model()
    # encode 返回 numpy array，转为 Python 列表
    embedding = model.encode(text)
    if isinstance(embedding, np.ndarray):
        return embedding.tolist()
    return list(embedding)