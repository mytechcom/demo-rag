# Ollama 服务地址与模型名称
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"           # 对话模型，可根据实际修改
EMBED_MODEL_NAME = "nomic-embed-text"  # 嵌入模型
# 本地嵌入模型路径（SentenceTransformer 可加载的路径或模型名）
LOCAL_EMBED_MODEL_PATH = "E:/zxc/llm/model/m3e-base"
# ChromaDB 持久化目录与集合名
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rag_demo"

# 简单切分参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# 父子切分参数
PARENT_CHUNK_SIZE = 1000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 50

# 语义切分参数
SEMANTIC_SIMILARITY_THRESHOLD = 0.7
MAX_SEMANTIC_CHUNK_SIZE = 800