import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import json
from dataclasses import dataclass, asdict


os.environ["OPENAI_API_KEY"] = "sk-25d23ee8f3a94186a9cb2bd9ddde85b1"
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"

PDF_PATH = "data/GBT+44510-2024.pdf"
DB_DIR_SMALL = "vectorstores/ev_small_chunks"
DB_DIR_LARGE = "vectorstores/ev_large_chunks"

def load_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 文件未找到：{pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"[INFO] Loaded {len(docs)} pages from PDF.")
    return docs

def split_documents(docs, chunk_size: int, chunk_overlap: int):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    split_docs = text_splitter.split_documents(docs)
    print(
        f"[INFO] Split into {len(split_docs)} chunks "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})."
    )
    return split_docs

def build_or_load_vectorstore(
    docs,
    persist_directory: str,
    chunk_size: int,
    chunk_overlap: int,
):

    embedding_model_name = "BAAI/bge-m3"

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"[INFO] Loading existing vectorstore from {persist_directory}")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        return vectordb

    print(f"[INFO] Building new vectorstore at {persist_directory}")
    split_docs = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    vectordb.persist()
    print(f"[INFO] Vectorstore built and persisted at {persist_directory}")
    return vectordb

def build_qa_chain(vectordb, model_name: str = "deepseek-chat"):

    llm = ChatOpenAI(
        model=model_name,
        temperature=0.0,
    )

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain

def ask_question(qa_chain, question: str, tag: str = ""):
    print(f"\n[{tag} QUESTION] {question}\n")
    result = qa_chain({"query": question})
    answer = result["result"]
    print(f"[{tag} ANSWER]")
    print(answer)
    print(f"\n[{tag} TOP-K CONTEXTS]")
    for i, doc in enumerate(result["source_documents"], start=1):
        print(f"--- Source #{i} (page: {doc.metadata.get('page', 'N/A')}) ---")
        print(doc.page_content[:200].replace("\n", " ") + "...")
    print("\n" + "=" * 80)
    return answer


def build_rag_eval_input(
    qa_chain,
    testset_path: str,
    output_path: str,
    max_samples: int = 50,
):

    if not os.path.exists(testset_path):
        raise FileNotFoundError(f"测试集文件未找到：{testset_path}")

    with open(testset_path, "r", encoding="utf-8") as f:
        test_items = json.load(f)

    eval_items = []

    test_items = test_items[:max_samples]

    print(f"[INFO] 从测试集选择 {len(test_items)} 条样本用于 RAG 评测输入生成。")

    for idx, item in enumerate(test_items, start=1):
        q = item["question"]
        gold_answer = item["ground_truth"]
        gold_context = item["context"]


        result = qa_chain({"query": q})
        rag_answer = result["result"]
        source_docs = result.get("source_documents", [])

        retrieved_contexts = [doc.page_content for doc in source_docs]

        eval_items.append(
            {
                "question": q,
                "ground_truth": gold_answer,
                "gold_context": gold_context,
                "answer": rag_answer,
                "retrieved_contexts": retrieved_contexts,
            }
        )

        print(f"[INFO] 已处理样本 {idx}/{len(test_items)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_items, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 评测输入已保存到：{output_path}")

def main():
    docs = load_pdf(PDF_PATH)

    vectordb_small = build_or_load_vectorstore(
        docs,
        persist_directory=DB_DIR_SMALL,
        chunk_size=200,
        chunk_overlap=50,
    )
    qa_small = build_qa_chain(vectordb_small)

    vectordb_large = build_or_load_vectorstore(
        docs,
        persist_directory=DB_DIR_LARGE,
        chunk_size=500,
        chunk_overlap=80,
    )
    qa_large = build_qa_chain(vectordb_large)

    questions: List[str] = [
        "新能源汽车高压系统在维护时需要遵守哪些安全操作步骤？",
        "动力电池在更换或拆装时有哪些关键技术要求？",
        "如果发生电池包热失控，按照规范应该如何处理？",
    ]

    for q in questions:
        print("\n" + "#" * 80)
        print("# 对比问题：", q)
        print("#" * 80)

        answer_small = ask_question(qa_small, q, tag="SMALL_CHUNK (size=200)")
        answer_large = ask_question(qa_large, q, tag="LARGE_CHUNK (size=500)")

    TESTSET_PATH = "outputs/generated_testset.json"

    build_rag_eval_input(
        qa_chain=qa_small,
        testset_path=TESTSET_PATH,
        output_path="data/rag_eval_input_small.json",
        max_samples=50,  # 可根据需要调整
    )

    build_rag_eval_input(
        qa_chain=qa_large,
        testset_path=TESTSET_PATH,
        output_path="data/rag_eval_input_large.json",
        max_samples=50,
    )

if __name__ == "__main__":
    main()
