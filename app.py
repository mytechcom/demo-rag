from embedding_client import get_embedding   # 使用本地模型
from ollama_client import chat               #
import os
import tempfile
from config import CHROMA_PERSIST_DIR, COLLECTION_NAME
from vector_store import VectorStore
from rag_engine import RAGEngine
import streamlit as st

# --------------------------- 初始化资源（持久化） ---------------------------
@st.cache_resource
def init_resources():
    vs = VectorStore(CHROMA_PERSIST_DIR, COLLECTION_NAME, get_embedding)
    engine = RAGEngine(vs, chat, get_embedding)
    return vs, engine

vector_store, rag_engine = init_resources()

st.set_page_config(page_title="RAG 教学 Demo", layout="wide")
st.title("📚 RAG 文档问答系统 (Ollama)")

# --------------------------- 侧边栏：文件管理 ---------------------------
with st.sidebar:
    st.header("📁 文件管理")
    uploaded_file = st.file_uploader(
        "上传文档 (txt, pdf, docx, md)",
        type=["txt", "pdf", "docx", "md"],
    )
    splitter_type = st.radio(
        "选择切分方式",
        ["simple", "parent_child", "semantic"],
        format_func=lambda x: {
            "simple": "简单切分",
            "parent_child": "父子切分",
            "semantic": "语义切分",
        }.get(x, x),
    )

    if st.button("🔨 建立索引", use_container_width=True):
        if uploaded_file is None:
            st.error("请先上传文件")
        else:
            # 将上传文件保存到临时目录
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                with st.spinner("正在处理文档并建立索引..."):
                    rag_engine.index_file(tmp_path, splitter_type)
                st.success(f"文件 `{uploaded_file.name}` 索引成功！")
            except Exception as e:
                st.error(f"索引失败: {str(e)}")
            finally:
                os.unlink(tmp_path)

    st.divider()
    st.subheader("已索引文件")
    sources = vector_store.get_sources()
    if sources:
        for src in sources:
            col1, col2 = st.columns([4, 1])
            col1.write(f"📄 {src}")
            if col2.button("删除", key=f"del_{src}"):
                vector_store.delete_by_source(src)
                st.rerun()
    else:
        st.info("暂无索引文件")

# --------------------------- 主界面：问答 ---------------------------
st.subheader("💬 知识库问答")
question = st.text_input("请输入你的问题", placeholder="例如：文档的主要内容是什么？")
top_k = st.slider("检索文档块数量", 1, 10, 4)

if st.button("🚀 提问", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("请输入问题")
    elif not vector_store.get_sources():
        st.warning("请先索引文档")
    else:
        with st.spinner("正在思考..."):
            try:
                result = rag_engine.query(question, top_k=top_k)
                st.markdown("### 📝 回答")
                st.write(result["answer"])

                st.markdown("### 📖 参考来源")
                for i, src in enumerate(result["sources"], 1):
                    with st.expander(f"来源 {i}: {src['source']} (相似度: {src.get('score', 'N/A')})"):
                        st.text(src["text"])
            except Exception as e:
                st.error(f"查询失败: {str(e)}")