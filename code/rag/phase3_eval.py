from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from env_utils import OPENAI_BASE_URL, OPENAI_MODEL


@dataclass(frozen=True)
class EvalSample:
    question: str
    ground_truth: str
    answer: str
    retrieved_contexts: List[str]
    q_type: str


@dataclass(frozen=True)
class EvalConfig:
    long_context_threshold_chars: int = 1200
    hard_negative_top_n: int = 3


class HardNegativeOutput(BaseModel):
    hard_negatives: List[str] = Field(default_factory=list)


class JudgeOutput(BaseModel):
    score: float
    verdict: str
    reason: str


class Phase3Evaluator:
    def __init__(self, config: EvalConfig | None = None):
        self.config = config or EvalConfig()
        self.judge_llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.0,
            openai_api_base=OPENAI_BASE_URL,
        )
        self.hard_negative_parser = PydanticOutputParser(
            pydantic_object=HardNegativeOutput
        )
        self.judge_parser = PydanticOutputParser(pydantic_object=JudgeOutput)

    def classify_layer(self, sample: EvalSample) -> str:
        if sample.q_type in {"fact", "multi-hop", "negative"}:
            return sample.q_type
        joined = "\n".join(sample.retrieved_contexts)
        if len(joined) >= self.config.long_context_threshold_chars:
            return "long_context"
        return "fact"

    def mine_hard_negatives(
        self,
        question: str,
        retrieved_contexts: List[str],
    ) -> List[str]:
        if not retrieved_contexts:
            return []

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are generating hard negatives for retrieval evaluation. "
                    "Select contexts that are topically related but likely misleading. "
                    "Return strict JSON object with key hard_negatives. "
                    "{format_instructions}",
                ),
                (
                    "human",
                    "Question:\n{question}\n\n"
                    "Candidate Contexts:\n{contexts}\n\n"
                    "Select up to {top_n} misleading contexts.",
                ),
            ]
        )
        contexts = "\n\n---\n\n".join(retrieved_contexts)
        response = self.judge_llm.invoke(
            prompt.format_messages(
                question=question,
                contexts=contexts,
                top_n=self.config.hard_negative_top_n,
                format_instructions=self.hard_negative_parser.get_format_instructions(),
            )
        )
        parsed = self.hard_negative_parser.parse(response.content)
        return [str(item) for item in parsed.hard_negatives if str(item).strip()]

    def _run_judge(
        self,
        role: str,
        question: str,
        answer: str,
        ground_truth: str,
        retrieved_contexts: List[str],
        hard_negatives: List[str],
    ) -> Dict[str, Any]:
        role_guidance = {
            "retriever": (
                "Focus on whether retrieved evidence is relevant and sufficient."
            ),
            "generator": (
                "Focus on factual consistency between answer and evidence."
            ),
            "safety": (
                "Focus on hallucination risk, unsupported safety claims, and uncertainty handling."
            ),
            "meta": (
                "Aggregate retriever/generator/safety views into final confidence."
            ),
        }
        if role not in role_guidance:
            raise ValueError(f"Unknown judge role: {role}")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a strict RAG judge. "
                    f"{role_guidance[role]} "
                    "Return strict JSON object with fields: "
                    "score (0~1 float), verdict (PASS/RETRY), reason (string). "
                    "{format_instructions}",
                ),
                (
                    "human",
                    "Question:\n{question}\n\n"
                    "Answer:\n{answer}\n\n"
                    "Ground Truth:\n{ground_truth}\n\n"
                    "Retrieved Contexts:\n{contexts}\n\n"
                    "Hard Negatives:\n{hard_negatives}",
                ),
            ]
        )
        contexts = "\n\n---\n\n".join(retrieved_contexts)
        negatives = "\n\n---\n\n".join(hard_negatives) if hard_negatives else ""
        response = self.judge_llm.invoke(
            prompt.format_messages(
                question=question,
                answer=answer,
                ground_truth=ground_truth,
                contexts=contexts,
                hard_negatives=negatives,
                format_instructions=self.judge_parser.get_format_instructions(),
            )
        )
        parsed = self.judge_parser.parse(response.content)
        return {
            "score": float(parsed.score),
            "verdict": str(parsed.verdict),
            "reason": str(parsed.reason),
        }

    def evaluate_sample(self, sample: EvalSample) -> Dict[str, Any]:
        layer = self.classify_layer(sample)
        hard_negatives = self.mine_hard_negatives(
            question=sample.question,
            retrieved_contexts=sample.retrieved_contexts,
        )
        retriever_judge = self._run_judge(
            role="retriever",
            question=sample.question,
            answer=sample.answer,
            ground_truth=sample.ground_truth,
            retrieved_contexts=sample.retrieved_contexts,
            hard_negatives=hard_negatives,
        )
        generator_judge = self._run_judge(
            role="generator",
            question=sample.question,
            answer=sample.answer,
            ground_truth=sample.ground_truth,
            retrieved_contexts=sample.retrieved_contexts,
            hard_negatives=hard_negatives,
        )
        safety_judge = self._run_judge(
            role="safety",
            question=sample.question,
            answer=sample.answer,
            ground_truth=sample.ground_truth,
            retrieved_contexts=sample.retrieved_contexts,
            hard_negatives=hard_negatives,
        )
        meta_judge = self._run_judge(
            role="meta",
            question=sample.question,
            answer=sample.answer,
            ground_truth=sample.ground_truth,
            retrieved_contexts=sample.retrieved_contexts,
            hard_negatives=hard_negatives,
        )
        return {
            "layer": layer,
            "hard_negatives": hard_negatives,
            "judges": {
                "retriever": retriever_judge,
                "generator": generator_judge,
                "safety": safety_judge,
                "meta": meta_judge,
            },
        }


def load_generated_testset(path: Path, limit: int) -> List[EvalSample]:
    if not path.exists():
        raise FileNotFoundError(f"Missing testset: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    samples: List[EvalSample] = []
    for item in data[:limit]:
        samples.append(
            EvalSample(
                question=item["question"],
                ground_truth=item["ground_truth"],
                answer=item["ground_truth"],
                retrieved_contexts=[item["context"]],
                q_type=item.get("q_type", "fact"),
            )
        )
    return samples
