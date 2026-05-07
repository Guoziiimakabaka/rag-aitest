from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from fastapi import FastAPI
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from env_utils import HF_HOME, OPENAI_BASE_URL, OPENAI_MODEL, RERANKER_MODEL
from phase3_eval import EvalSample, Phase3Evaluator


@dataclass(frozen=True)
class RetrievalConfig:
    pdf_path: Path
    vectorstore_path: Path
    embedding_model: str = "BAAI/bge-m3"
    dense_k: int = 12
    bm25_k: int = 12
    fused_dense_weight: float = 0.6
    fused_bm25_weight: float = 0.4
    reranker_model: str = RERANKER_MODEL
    reranker_top_n: int = 8
    hyde_enabled: bool = True
    multi_query_enabled: bool = True
    multi_query_count: int = 3


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    use_hyde: bool = True
    use_multi_query: bool = True


class RetrievedDoc(BaseModel):
    rank: int
    score: float
    page: int | None
    content: str


class RetrieveResponse(BaseModel):
    expanded_queries: List[str]
    documents: List[RetrievedDoc]


class AskResponse(BaseModel):
    answer: str
    expanded_queries: List[str]
    documents: List[RetrievedDoc]


class ReflectionRequest(QueryRequest):
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_retries: int = Field(default=1, ge=0, le=3)


class ReflectionResponse(BaseModel):
    answer: str
    confidence: float
    needs_retry: bool
    retries_used: int
    expanded_queries: List[str]
    documents: List[RetrievedDoc]
    judge_reason: str


class Phase3EvalRequest(BaseModel):
    question: str = Field(min_length=1)
    ground_truth: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    retrieved_contexts: List[str] = Field(min_length=1)
    q_type: str = Field(default="fact")


class JudgeResult(BaseModel):
    score: float
    verdict: str
    reason: str


class Phase3EvalResponse(BaseModel):
    layer: str
    hard_negatives: List[str]
    retriever_judge: JudgeResult
    generator_judge: JudgeResult
    safety_judge: JudgeResult
    meta_judge: JudgeResult


class HybridRagService:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.0,
            openai_api_base=OPENAI_BASE_URL,
        )
        cache_folder = HF_HOME if HF_HOME else None
        self.cross_encoder = CrossEncoder(
            config.reranker_model,
            cache_folder=cache_folder,
        )
        self.documents = self._load_documents(config.pdf_path)
        self.ensemble_retriever = self._build_hybrid_retriever()

    @staticmethod
    def _load_documents(pdf_path: Path):
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing PDF file: {pdf_path}")
        loader = PyPDFLoader(str(pdf_path))
        return loader.load()

    def _build_hybrid_retriever(self) -> EnsembleRetriever:
        model_kwargs = {}
        if HF_HOME:
            model_kwargs["cache_folder"] = HF_HOME
            model_kwargs["model_kwargs"] = {"local_files_only": True}
        embedding = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            **model_kwargs,
        )
        vector_db = Chroma(
            persist_directory=str(self.config.vectorstore_path),
            embedding_function=embedding,
        )
        dense_retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.dense_k},
        )
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = self.config.bm25_k
        return EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[
                self.config.fused_dense_weight,
                self.config.fused_bm25_weight,
            ],
        )

    def _build_multi_queries(self, question: str, use_multi_query: bool) -> List[str]:
        if not use_multi_query or not self.config.multi_query_enabled:
            return [question]

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You rewrite a user query into diverse search queries. "
                    "Return strict JSON list of strings only.",
                ),
                (
                    "human",
                    "Original question:\n{question}\n\n"
                    "Generate {count} semantically diverse retrieval queries.",
                ),
            ]
        )
        message = prompt.format_messages(
            question=question,
            count=self.config.multi_query_count,
        )
        response = self.llm.invoke(message)
        queries = json.loads(response.content)
        if not isinstance(queries, list):
            raise TypeError("Multi-query response must be a JSON list.")
        normalized = [str(item).strip() for item in queries if str(item).strip()]
        if not normalized:
            raise ValueError("Multi-query generation returned empty queries.")
        return [question] + normalized

    def _build_hyde_query(self, question: str, use_hyde: bool) -> str | None:
        if not use_hyde or not self.config.hyde_enabled:
            return None

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Write a concise hypothetical answer passage that may contain "
                    "helpful retrieval terms for the question.",
                ),
                ("human", "Question:\n{question}"),
            ]
        )
        response = self.llm.invoke(prompt.format_messages(question=question))
        text = response.content.strip()
        if not text:
            raise ValueError("HyDE generation returned empty text.")
        return text

    @staticmethod
    def _deduplicate_docs(docs: List[Document]) -> List[Document]:
        unique = {}
        for doc in docs:
            key = (
                doc.metadata.get("source"),
                doc.metadata.get("page"),
                doc.page_content.strip(),
            )
            if key not in unique:
                unique[key] = doc
        return list(unique.values())

    def _rerank(self, question: str, docs: List[Document]):
        pairs = [[question, doc.page_content] for doc in docs]
        scores = self.cross_encoder.predict(pairs)
        scored = list(zip(docs, scores, strict=True))
        scored.sort(key=lambda item: float(item[1]), reverse=True)
        return scored[: self.config.reranker_top_n]

    def retrieve(self, question: str, use_hyde: bool, use_multi_query: bool):
        expanded_queries = self._build_multi_queries(question, use_multi_query)
        hyde_query = self._build_hyde_query(question, use_hyde)
        if hyde_query:
            expanded_queries.append(hyde_query)
        docs = []
        for query in expanded_queries:
            docs.extend(self.ensemble_retriever.invoke(query))
        docs = self._deduplicate_docs(docs)
        reranked = self._rerank(question, docs)
        return expanded_queries, reranked

    def ask(self, question: str, use_hyde: bool, use_multi_query: bool):
        expanded_queries, reranked = self.retrieve(
            question=question,
            use_hyde=use_hyde,
            use_multi_query=use_multi_query,
        )
        context = "\n\n---\n\n".join([doc.page_content for doc, _ in reranked])
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an assistant for standard-based maintenance QA. "
                    "Answer only from provided context. If context is insufficient, "
                    "explicitly say insufficient evidence.",
                ),
                (
                    "human",
                    "Question:\n{question}\n\nContext:\n{context}",
                ),
            ]
        )
        response = self.llm.invoke(
            prompt.format_messages(question=question, context=context)
        )
        return response.content, expanded_queries, reranked

    def _judge_answer(
        self,
        question: str,
        answer: str,
        reranked,
    ) -> tuple[float, bool, str]:
        context = "\n\n---\n\n".join([doc.page_content for doc, _ in reranked])
        judge_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a strict RAG evaluator. "
                    "Evaluate whether the answer is fully supported by evidence. "
                    "Return strict JSON with fields: "
                    "confidence (0~1 float), needs_retry (true/false), reason (string).",
                ),
                (
                    "human",
                    "Question:\n{question}\n\n"
                    "Answer:\n{answer}\n\n"
                    "Evidence:\n{context}",
                ),
            ]
        )
        response = self.llm.invoke(
            judge_prompt.format_messages(
                question=question,
                answer=answer,
                context=context,
            )
        )
        parsed = json.loads(response.content)
        confidence = float(parsed["confidence"])
        needs_retry = bool(parsed["needs_retry"])
        reason = str(parsed["reason"])
        return confidence, needs_retry, reason

    def ask_with_reflection(
        self,
        question: str,
        use_hyde: bool,
        use_multi_query: bool,
        confidence_threshold: float,
        max_retries: int,
    ) -> tuple[str, float, bool, int, List[str], list, str]:
        retries_used = 0
        answer, expanded_queries, reranked = self.ask(
            question=question,
            use_hyde=use_hyde,
            use_multi_query=use_multi_query,
        )
        confidence, needs_retry, reason = self._judge_answer(
            question=question,
            answer=answer,
            reranked=reranked,
        )

        while (
            retries_used < max_retries
            and needs_retry
            and confidence < confidence_threshold
        ):
            retries_used += 1
            answer, expanded_queries, reranked = self.ask(
                question=question,
                use_hyde=True,
                use_multi_query=True,
            )
            confidence, needs_retry, reason = self._judge_answer(
                question=question,
                answer=answer,
                reranked=reranked,
            )

        return (
            answer,
            confidence,
            needs_retry,
            retries_used,
            expanded_queries,
            reranked,
            reason,
        )


def _to_response_docs(reranked) -> List[RetrievedDoc]:
    docs: List[RetrievedDoc] = []
    for idx, (doc, score) in enumerate(reranked, start=1):
        docs.append(
            RetrievedDoc(
                rank=idx,
                score=float(score),
                page=doc.metadata.get("page"),
                content=doc.page_content,
            )
        )
    return docs


def build_service() -> HybridRagService:
    root = Path(__file__).resolve().parents[2]
    config = RetrievalConfig(
        pdf_path=root / "code" / "rag" / "data" / "GBT+44510-2024.pdf",
        vectorstore_path=root / "code" / "rag" / "vectorstores" / "ev_large_chunks",
    )
    return HybridRagService(config=config)


service = build_service()
phase3_evaluator = Phase3Evaluator()
app = FastAPI(title="RAG-Eye Phase1 API", version="0.1.0")


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: QueryRequest) -> RetrieveResponse:
    expanded_queries, reranked = service.retrieve(
        question=request.question,
        use_hyde=request.use_hyde,
        use_multi_query=request.use_multi_query,
    )
    return RetrieveResponse(
        expanded_queries=expanded_queries,
        documents=_to_response_docs(reranked),
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: QueryRequest) -> AskResponse:
    answer, expanded_queries, reranked = service.ask(
        question=request.question,
        use_hyde=request.use_hyde,
        use_multi_query=request.use_multi_query,
    )
    return AskResponse(
        answer=answer,
        expanded_queries=expanded_queries,
        documents=_to_response_docs(reranked),
    )


@app.post("/ask_with_reflection", response_model=ReflectionResponse)
def ask_with_reflection(request: ReflectionRequest) -> ReflectionResponse:
    (
        answer,
        confidence,
        needs_retry,
        retries_used,
        expanded_queries,
        reranked,
        judge_reason,
    ) = service.ask_with_reflection(
        question=request.question,
        use_hyde=request.use_hyde,
        use_multi_query=request.use_multi_query,
        confidence_threshold=request.confidence_threshold,
        max_retries=request.max_retries,
    )
    return ReflectionResponse(
        answer=answer,
        confidence=confidence,
        needs_retry=needs_retry,
        retries_used=retries_used,
        expanded_queries=expanded_queries,
        documents=_to_response_docs(reranked),
        judge_reason=judge_reason,
    )


@app.post("/phase3/evaluate_sample", response_model=Phase3EvalResponse)
def evaluate_phase3_sample(request: Phase3EvalRequest) -> Phase3EvalResponse:
    sample = EvalSample(
        question=request.question,
        ground_truth=request.ground_truth,
        answer=request.answer,
        retrieved_contexts=request.retrieved_contexts,
        q_type=request.q_type,
    )
    result = phase3_evaluator.evaluate_sample(sample)
    return Phase3EvalResponse(
        layer=result["layer"],
        hard_negatives=result["hard_negatives"],
        retriever_judge=JudgeResult(**result["judges"]["retriever"]),
        generator_judge=JudgeResult(**result["judges"]["generator"]),
        safety_judge=JudgeResult(**result["judges"]["safety"]),
        meta_judge=JudgeResult(**result["judges"]["meta"]),
    )
