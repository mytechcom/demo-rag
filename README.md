# 📚 RAG 教学实验 Demo

一个基于 Python 的 **RAG（检索增强生成）** 教学演示系统，支持多格式文档管理、三种文本切分策略，并使用本地向量模型与 Ollama 部署的大模型进行问答。提供 Streamlit 图形界面，适合学习 RAG 原理与工程实践。

---

## ✨ 功能特性

- **文档上传与索引**：支持 `txt`、`pdf`、`docx`、`md` 格式文档的自动解析与向量化存储。
- **三种文本切分方式**：
  1. **简单切分**：固定大小 + 重叠的文本块。
  2. **父子切分**：父块提供更大上下文，子块用于精确检索。
  3. **语义切分**：基于相邻句子嵌入相似度的智能断句，保持语义完整性。
- **本地嵌入模型**：使用 `sentence-transformers` 加载本地或 HuggingFace 模型（示例：`m3e-base`）生成高质量向量，无需额外服务。
- **Ollama 对话模型**：通过 Ollama 部署的 LLM（如 `qwen2.5:7b`）生成自然语言回答。
- **ChromaDB 向量存储**：支持持久化、按源文件管理、高效相似度检索。
- **参考来源展示**：回答时自动列出检索到的原文片段及相似度分数，增强可解释性。
- **模块化工程结构**：高内聚低耦合，各组件可独立替换或扩展。

---

## 📁 项目结构

```
rag_demo/
├── config.py             # 所有可配置参数（模型路径、切分参数等）
├── ollama_client.py      # Ollama API 对话客户端
├── embedding_client.py   # 本地 SentenceTransformer 嵌入模型客户端
├── document_loader.py    # 多格式文档加载器
├── text_splitter.py      # 三种切分器实现
├── vector_store.py       # ChromaDB 向量数据库封装
├── rag_engine.py         # RAG 核心逻辑（索引 + 检索 + 答案生成）
├── app.py                # Streamlit 前端界面
├── requirements.txt      # Python 依赖列表
└── README.md             # 本说明文档
```

---

## 🚀 快速开始

### 1. 环境准备

#### 1.1 创建 Conda 虚拟环境（推荐）

```bash
conda create -n rag-demo python=3.10 -y
conda activate rag-demo
```

#### 1.2 安装 Python 依赖

```bash
pip install -r requirements.txt
```

> 注意：`sentence-transformers` 会自动安装 PyTorch 等依赖，请确保网络通畅。

### 2. 启动 Ollama 服务并下载模型

#### 2.1 安装 Ollama
访问 [Ollama 官网](https://ollama.com/) 下载并安装对应操作系统版本。

#### 2.2 启动服务并拉取模型
```bash
# 启动 Ollama 后台服务
ollama serve

# 下载对话模型（可根据硬件调整，示例为 qwen2.5:7b）
ollama pull qwen2.5:7b
```

### 3. 配置本地嵌入模型

编辑 `config.py`，将 `LOCAL_EMBED_MODEL_PATH` 改为你本地的模型路径（支持 HuggingFace 模型名称或本地文件夹）：

```python
# 示例：使用本地下载的 m3e-base 模型
LOCAL_EMBED_MODEL_PATH = "/data/model/m3e-base"

# 也可以直接使用 HuggingFace 模型名（会自动下载）
# LOCAL_EMBED_MODEL_PATH = "moka-ai/m3e-base"
```

确保该路径下包含完整的模型文件（或通过联网自动拉取）。

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器将自动打开 `http://localhost:8501`，即可使用 RAG 问答系统。

---

## 📖 使用指南

1. **上传文档**  
   在左侧边栏点击“Browse files”上传一个或多个文档（支持 txt、pdf、docx、md）。

2. **选择切分方式**  
   根据文档特点选择：
   - **简单切分**：通用文本，快速。
   - **父子切分**：需要更大上下文时。
   - **语义切分**：保留段落语义，避免截断。

3. **点击“建立索引”**  
   系统将自动加载文档、切分、生成向量并存入 ChromaDB。  
   已索引的文件会显示在下方列表中，可随时删除。

4. **提问**  
   在主界面输入问题（如“本文的主要观点是什么？”），调整检索块数量（推荐 3~5），点击“提问”。  
   答案下方会展开显示所有参考来源及其相似度分数，你可以直接查看原文。

---

## ⚙️ 高级配置

所有参数均集中在 `config.py` 中：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `OLLAMA_BASE_URL` | Ollama 服务地址 | `http://localhost:11434` |
| `MODEL_NAME` | 对话模型名称 | `qwen2.5:7b` |
| `LOCAL_EMBED_MODEL_PATH` | 本地嵌入模型路径 | `E:/zxc/llm/model/m3e-base` |
| `CHROMA_PERSIST_DIR` | 向量库持久化路径 | `./chroma_db` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 简单切分参数 | 500 / 100 |
| `PARENT_CHUNK_SIZE` 等 | 父子切分参数 | 1000 / 200 / 300 / 50 |
| `SEMANTIC_SIMILARITY_THRESHOLD` | 语义切分相似阈值 | 0.7 |

可根据自己的文档长度、模型上下文窗口等调整。

---

## ❓ 常见问题

**Q：启动时提示“ModuleNotFoundError: No module named 'xxx'”**  
A：确保已激活 conda 环境，并执行 `pip install -r requirements.txt`。

**Q：Ollama 无法连接**  
A：检查 Ollama 服务是否启动（`ollama serve`）；若端口非默认，修改 `config.py` 中的 `OLLAMA_BASE_URL`。

**Q：嵌入模型加载失败**  
A：确认模型路径正确，且文件夹内包含 `config.json` 等必要文件。若使用 HuggingFace 模型名，请检查网络连接。

**Q：回答不准确**  
A：尝试调整切分方式（长文档推荐父子切分）、增大 `top_k` 值、或检查文档是否被正确索引。

---

## 📄 许可

本项目仅用于教学与学习目的。如使用第三方模型，请遵守其原始许可协议。

---

## 🙌 致谢

- [LangChain](https://github.com/langchain-ai/langchain) 启发切分器设计
- [ChromaDB](https://www.trychroma.com/) 提供高效向量存储
- [Ollama](https://ollama.com/) 便捷的本地大模型部署
- [Streamlit](https://streamlit.io/) 快速构建数据界面
