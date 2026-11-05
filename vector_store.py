"""
用于向量存储,包含加载指定目录下的所有文件，通过校验md5判断文件是否已经存入数据库
未存入数据库的进行文本分割后存入数据库
VectorStoreService是对象，需要创建
可以使用load_document加载数据
可以使用get_retriever来检索数据库内容

文件加载（txt/pdf/csv）	✅ 自动编码探测 + metadata 标注
按类型切片（chunking）	✅ txt 400字 / pdf 1000字 / csv 不切
向量入库（Chroma）	✅ 分批 20 条，MD5 增量跳过
    1.纯语义检索	✅ get_retriever()
    2.BM25+向量混合检索	✅ get_hybrid_retriever()（但权重未用）
    3.带过滤检索	✅ get_retriever_with_filter()
"""
import os
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, csv_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from langchain_core.documents import Document


class VectorStoreService(object):
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),  # chroma的存储路径
        )

        # ----- 按文件类型创建不同的文本分割器（配置来自 chroma.yml） -----
        chunk_cfg = chroma_conf.get("chunk_config", {})

        # .txt 分割器（优先按标题切，保持章节语义）
        txt_cfg = chunk_cfg.get("txt", {})
        self.txt_splitter = RecursiveCharacterTextSplitter(
            chunk_size=txt_cfg.get("chunk_size", 400),
            chunk_overlap=txt_cfg.get("chunk_overlap", 50),
            separators=txt_cfg.get("separators", ["\n\n## ", "\n\n### ", "\n\n", "\n", "。", "！", "？"]),
            length_function=len,
        )

        # .pdf 分割器（段落更长，少切）
        pdf_cfg = chunk_cfg.get("pdf", {})
        self.pdf_splitter = RecursiveCharacterTextSplitter(
            chunk_size=pdf_cfg.get("chunk_size", 1000),
            chunk_overlap=pdf_cfg.get("chunk_overlap", 100),
            separators=pdf_cfg.get("separators", ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", "，", ","]),
            length_function=len,
        )

        # ----- BM25 关键词检索器（懒初始化） -----
        self._bm25_retriever = None

    def _ensure_bm25(self):
        """确保 BM25 检索器已构建"""
        if self._bm25_retriever is not None:
            return
        all_docs_data = self.vector_store.get()  # 从 Chroma 向量库中拉取全部已存储的文档及其元数据
        docs = all_docs_data.get("documents") or []  # 从字典中取出 documents（文本列表）和 metadatas（元数据列表）。
        metas = all_docs_data.get("metadatas") or []
        if docs:
            self._bm25_retriever = BM25Retriever.from_texts(
                docs, metadatas=metas
            )
            self._bm25_retriever.k = chroma_conf["k"]
        else:
            self._bm25_retriever = None

    def get_retriever(self):
        """返回向量检索器（纯语义）"""
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def get_retriever_with_filter(self, filter_dict: dict):
        """返回带 metadata 过滤的检索器

        Args:
            filter_dict: Chroma 过滤条件，如 {"source_file": "error_codes.csv"}

        用法:
            retriever = vs.get_retriever_with_filter({"error_prefix": "CE"})
            results = retriever.invoke("夹爪")
        """
        return self.vector_store.as_retriever(
            search_kwargs={"k": chroma_conf["k"], "filter": filter_dict}
        )

    def get_hybrid_retriever(self):
        """返回一个可调用的混合检索函数，合并 BM25 + 向量结果"""
        self._ensure_bm25()

        def hybrid_invoke(query: str):
            # 1. 向量检索
            vector_docs = self.vector_store.similarity_search(query, k=chroma_conf["k"])
            # 2. BM25 关键词检索
            bm25_docs = self._bm25_retriever.invoke(query) if self._bm25_retriever else []
            # 3. 按 page_content 去重合并（BM25 结果放在前面，因为对精确代码匹配更好）
            seen = set()
            merged = []
            for doc in bm25_docs + vector_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    merged.append(doc)
            return merged[:chroma_conf["k"]]

        return hybrid_invoke

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库   不需要指定文件，文件夹写死了
        要计算文件的md5，加载过的文件不处理了
        :return:
        """

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()  # 去掉字符串 line 两端的所有空白字符（空格、换行符、制表符等）
                    if line == md5_for_check:
                        return True

                return False

        def save_md5_hex(md5_for_check: str):
            """保存 MD5 校验值"""
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str) -> list[Document]:
            """根据文件后缀分发加载器（加载器会附带 metadata）"""
            if read_path.endswith(".txt"):
                return txt_loader(read_path)

            if read_path.endswith(".pdf"):
                return pdf_loader(read_path)

            if read_path.endswith(".csv"):
                return csv_loader(read_path)

            return []

        def split_documents_by_type(docs: list, read_path: str) -> list:
            """根据文件后缀选择分割策略（metadata 会随分割自动继承）"""
            if read_path.endswith(".csv"):
                # CSV 每行已是独立完整信息，不再切分
                return docs

            if read_path.endswith(".pdf"):
                # PDF 用较大的分块（长段落）
                return self.pdf_splitter.split_documents(docs)

            # .txt 及其他用 TXT 分割器（优先按标题分割）
            return self.txt_splitter.split_documents(docs)

        # 检索指定目录下的文件类型获取绝对路径
        allow_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"])
        )
        for path in allow_files_path:
            md5 = get_file_md5_hex(path)  # 检验md5判断是否导入过
            if check_md5_hex(md5):
                logger.info(f"[加载知识库]{path}已经存在知识库[跳过]")
                continue
            # 此前没有导入，进行分割 导入向量库
            try:
                documemt = get_file_documents(path)
                if not documemt:
                    logger.warning(f"[加载知识库]{path}文本内无内容")
                    continue

                # 按文件类型选择分割器（metadata 自动保留）
                split_document = split_documents_by_type(documemt, path)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}无法分割出有效内容")
                    continue

                # DashScope Embedding API 限制每批 ≤ 20 条，需要分批提交
                batch_size = 20
                for i in range(0, len(split_document), batch_size):
                    batch = split_document[i : i + batch_size]
                    self.vector_store.add_documents(batch)
                    logger.info(f"{path} 分批入库 [{i+1}-{i+len(batch)}/{len(split_document)}]")

                save_md5_hex(md5)  # 保存文件的md5
                logger.info(f"{path}加载知识库成功")
            except Exception as e:
                # exc_info=True 的作用是：把完整的异常堆栈信息（traceback）一起记录到日志里。
                logger.error(f"{path}加载知识库失败：{str(e)}", exc_info=True)

        # 加载完成后，重置 BM25（下次 get_hybrid_retriever 会重新构建）
        self._bm25_retriever = None


if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()  # 加载data里面所有的文件

    # 使用混合检索（BM25 + 向量）
    hybrid_retriever = vs.get_hybrid_retriever()
    res = hybrid_retriever("CE0502")
    print("=" * 60)
    for r in res:
        print(f"📄 [{r.metadata.get('source_file', '?')}] "
              f"error_code={r.metadata.get('error_code', 'N/A')}")
        print(r.page_content[:150])
        print("-" * 40)

    print("\n🎯 带 metadata 过滤的检索示例:")
    retriever = vs.get_retriever_with_filter({"source_file": "error_codes.csv"})
    filtered = retriever.invoke("CE0502")
    for r in filtered:
        print(f"  [{r.metadata.get('error_code', '?')}] {r.page_content[:100]}")
