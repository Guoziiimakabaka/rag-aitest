import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm
from langchain_openai import ChatOpenAI


OPENAI_API_KEY = "sk-25d23ee8f3a94186a9cb2bd9ddde85b1"
OPENAI_BASE_URL = "https://api.deepseek.com/v1"
JUDGE_MODEL = "deepseek-chat"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


@dataclass
class EvalSample:

    question: str
    ground_truth: str
    gold_context: str
    answer: str
    retrieved_contexts: List[str]

@dataclass
class EvalResult:
    question: str
    ground_truth: str
    answer: str
    context_recall: float
    context_precision: float
    faithfulness: float
    answer_relevance: float
    context_recall_explanation: str
    context_precision_explanation: str
    faithfulness_explanation: str
    answer_relevance_explanation: str


def get_judge_llm() -> ChatOpenAI:
    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_BASE_URL,
    )
    return llm


def _parse_score_from_llm(content: str) -> (float, str):
    score = 0.0
    explanation = content.strip()
    lines = content.splitlines()
    for line in lines:
        if "SCORE" in line.upper():
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    score_val = float(parts[1].strip())
                    # 限制在 [0,1]
                    score = max(0.0, min(1.0, score_val))
                except Exception:
                    pass
    return score, explanation


def judge_context_recall(
    llm: ChatOpenAI,
    sample: EvalSample,
) -> (float, str):

    retrieved_text = "\n\n---\n\n".join(sample.retrieved_contexts)

    system_prompt = """你是一个评测专家，负责评估RAG系统的“上下文召回率（Context Recall）”。

给定：
- 用户问题
- 黄金参考上下文（gold_context）：回答该问题所必需且充分的标准文本片段
- 模型实际检索到的上下文集合（retrieved_contexts）

请判断：模型检索到的上下文中，是否覆盖了“回答问题所需的关键信息”。

评分规则（0.0 ~ 1.0）：
- 1.0：检索到的上下文清楚地包含了回答问题所需的全部关键信息，与黄金上下文高度重合。
- 0.7：大部分关键点都能在检索到的上下文中找到，只有少量细节缺失。
- 0.4：只覆盖了部分关键信息，缺少较多关键条款或数值。
- 0.1：几乎没有检索到有用信息，与黄金上下文几乎无关。
- 0.0：完全未覆盖任何关键内容。

输出格式：
SCORE: <0~1之间的小数>
EXPLANATION: <用2~4句话解释你的判断依据>"""

    user_prompt = f"""[问题]
{sample.question}

[黄金参考上下文 gold_context]
{sample.gold_context}

[实际检索到的上下文 retrieved_contexts]
{retrieved_text}

请按照评分规则给出Context Recall的分数，并解释原因。"""

    resp = llm.invoke(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}]
    )
    return _parse_score_from_llm(resp.content)

def judge_context_precision(
    llm: ChatOpenAI,
    sample: EvalSample,
) -> (float, str):

    retrieved_text = "\n\n---\n\n".join(sample.retrieved_contexts)

    system_prompt = """你是一个评测专家，负责评估RAG系统的“上下文精准度（Context Precision）”。

给定：
- 用户问题
- 模型实际检索到的上下文集合（retrieved_contexts）

请判断：在这些检索到的文本中，有多大比例是真正用来回答该问题的有用信息，而不是无关、冗余或噪音。

评分规则（0.0 ~ 1.0）：
- 1.0：几乎所有检索到的内容都与问题高度相关，且直接用于回答问题。
- 0.7：大部分内容与问题相关，但夹杂了一些无关或边缘信息。
- 0.4：有明显噪音，真正有用的部分不足一半。
- 0.1：极少部分内容与问题相关，大多数是无关信息。
- 0.0：完全无关或错误信息。

请根据整体情况综合给出一个0~1之间的得分，不需要精确比例，但要体现趋势。

输出格式：
SCORE: <0~1之间的小数>
EXPLANATION: <用2~4句话解释你的判断依据>"""

    user_prompt = f"""[问题]
{sample.question}

[实际检索到的上下文 retrieved_contexts]
{retrieved_text}

请按照评分规则给出Context Precision的分数，并解释原因。"""

    resp = llm.invoke(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}]
    )
    return _parse_score_from_llm(resp.content)

def judge_faithfulness(
    llm: ChatOpenAI,
    sample: EvalSample,
) -> (float, str):

    retrieved_text = "\n\n---\n\n".join(sample.retrieved_contexts)

    system_prompt = """你是一个评测专家，负责评估RAG系统回答的“信实度（Faithfulness）”。

给定：
- 用户问题
- 模型回答（answer）
- 模型检索到的上下文集合（retrieved_contexts）
- （仅供参考）一个标准答案 ground_truth

你的任务：只根据检索到的上下文，判断模型回答是否“有据可依”，有没有编造上下文中不存在的事实。

评分时：
- 不要求回答与ground_truth完全一致，只要回答内容能在retrieved_contexts中找到充分证据，就认为是“忠实的”。
- 如果回答中包含上下文没有提到的具体技术要求、数值、步骤、结论，则按幻觉处理，降低得分。

评分规则（0.0 ~ 1.0）：
- 1.0：回答中的关键事实都能在上下文中找到明确依据，未发现编造内容。
- 0.7：大部分内容有依据，但存在少量模糊推断或轻微扩展。
- 0.4：回答中有明显内容在上下文中找不到依据，但仍有部分正确。
- 0.1：回答大部分内容没有依据或明显与上下文冲突。
- 0.0：回答几乎完全脱离上下文或完全错误。

输出格式：
SCORE: <0~1之间的小数>
EXPLANATION: <用2~5句话解释你的判断依据，指出哪些内容有依据、哪些可能是幻觉>"""

    user_prompt = f"""[问题]
{sample.question}

[模型回答 answer]
{sample.answer}

[检索到的上下文 retrieved_contexts]
{retrieved_text}

[参考答案 ground_truth（仅供对比，不强制要求一致）]
{sample.ground_truth}

请基于检索到的上下文评估回答的Faithfulness，并打分。"""

    resp = llm.invoke(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}]
    )
    return _parse_score_from_llm(resp.content)

def judge_answer_relevance(
    llm: ChatOpenAI,
    sample: EvalSample,
) -> (float, str):

    retrieved_text = "\n\n---\n\n".join(sample.retrieved_contexts)

    system_prompt = """你是一个评测专家，负责评估RAG系统回答的“答案相关性（Answer Relevance）”。

给定：
- 用户问题（question）
- 模型回答（answer）
- 检索到的上下文（retrieved_contexts）
- 标准答案 ground_truth（代表问题希望得到的信息，但不要求完全一致）

你的任务：判断模型回答是否真正回答了问题的核心，是否聚焦关键点，是否存在严重跑题或答非所问。

评分规则（0.0 ~ 1.0）：
- 1.0：回答紧扣问题，覆盖了ground_truth中的关键点，没有明显跑题。
- 0.7：回答与问题基本相关，覆盖了部分关键点，但有遗漏或赘述。
- 0.4：回答只部分触及问题，很多重要点未提及，或夹杂较多无关内容。
- 0.1：回答与问题关联很弱，基本没有解决用户关切。
- 0.0：回答完全与问题无关，或者答非所问。

输出格式：
SCORE: <0~1之间的小数>
EXPLANATION: <用2~5句话解释你的判断依据，指出是否覆盖了问题关键点>"""

    user_prompt = f"""[问题 question]
{sample.question}

[模型回答 answer]
{sample.answer}

[检索到的上下文 retrieved_contexts]
{retrieved_text}

[标准答案 ground_truth]
{sample.ground_truth}

请评估该回答的Answer Relevance，并打分。"""

    resp = llm.invoke(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}]
    )
    return _parse_score_from_llm(resp.content)


def evaluate_samples(samples: List[EvalSample]) -> List[EvalResult]:
    llm = get_judge_llm()
    results: List[EvalResult] = []

    for sample in tqdm(samples, desc="Evaluating samples"):
        cr_score, cr_exp = judge_context_recall(llm, sample)
        cp_score, cp_exp = judge_context_precision(llm, sample)
        fa_score, fa_exp = judge_faithfulness(llm, sample)
        ar_score, ar_exp = judge_answer_relevance(llm, sample)

        res = EvalResult(
            question=sample.question,
            ground_truth=sample.ground_truth,
            answer=sample.answer,
            context_recall=cr_score,
            context_precision=cp_score,
            faithfulness=fa_score,
            answer_relevance=ar_score,
            context_recall_explanation=cr_exp,
            context_precision_explanation=cp_exp,
            faithfulness_explanation=fa_exp,
            answer_relevance_explanation=ar_exp,
        )
        results.append(res)
    return results

def summarize_results(results: List[EvalResult]) -> Dict[str, float]:
    if not results:
        return {}

    def avg(attr: str) -> float:
        vals = [getattr(r, attr) for r in results]
        return sum(vals) / len(vals)

    summary = {
        "avg_context_recall": avg("context_recall"),
        "avg_context_precision": avg("context_precision"),
        "avg_faithfulness": avg("faithfulness"),
        "avg_answer_relevance": avg("answer_relevance"),
    }
    return summary


def load_eval_samples_from_json(json_path: str) -> List[EvalSample]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[EvalSample] = []
    for item in data:
        samples.append(
            EvalSample(
                question=item["question"],
                ground_truth=item["ground_truth"],
                gold_context=item["gold_context"],
                answer=item["answer"],
                retrieved_contexts=item["retrieved_contexts"],
            )
        )
    return samples

def save_results(results: List[EvalResult], json_path: str, csv_path: str):
    dicts = [asdict(r) for r in results]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dicts, f, ensure_ascii=False, indent=2)
    df = pd.DataFrame(dicts)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    eval_input_path = "data/rag_eval_input_large.json"
    results_json = "outputs/large/rag_eval_results.json"
    results_csv = "outputs/large/rag_eval_results.csv"

    samples = load_eval_samples_from_json(eval_input_path)
    results = evaluate_samples(samples)
    save_results(results, results_json, results_csv)

    summary = summarize_results(results)
    print("==== Overall Summary ====")
    for k, v in summary.items():
        print(f"{k}: {v:.3f}")