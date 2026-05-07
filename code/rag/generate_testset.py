import os
import json
from typing import List, Literal
from dataclasses import dataclass, asdict

import pandas as pd
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from env_utils import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL



MODEL_NAME = OPENAI_MODEL

PDF_PATH = "data/GBT+44510-2024.pdf"
OUTPUT_JSON = "outputs/generated_testset.json"
OUTPUT_CSV = "outputs/generated_testset.csv"

TARGET_NUM_QUESTIONS = 60

NUM_Q_PER_CHUNK = 3


QuestionType = Literal["fact", "multi-hop", "negative"]

@dataclass
class QAItem:
    question: str
    ground_truth: str
    context: str
    q_type: QuestionType
    source_page: int


def load_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 文件未找到：{pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"[INFO] Loaded {len(docs)} pages from PDF.")
    return docs

def split_documents(docs, chunk_size: int = 800, chunk_overlap: int = 150):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap}).")
    return chunks


def get_llm():

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.2,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_BASE_URL,
    )
    return llm


SYSTEM_PROMPT = """你是一个专业的试题命制专家，长期为汽车维修企业、检测站和监管机构设计闭卷考试和在线测评题库。
你现在要根据给定的“新能源汽车维修维护技术要求”标准片段，为一个 RAG 问答系统生成高质量测试集。

总体原则：
1. 严格以【给定片段】为唯一依据，不能引入片段之外的知识或行业常识。
2. 所有问题都要贴近真实工作场景：假设提问者是“维修技师、质检工程师、安监人员”，他们问的问题是为了“按标准正确施工/验收/排查隐患”。
3. 输出的 JSON 中，每道题必须包含：
   - type: "fact" | "multi-hop" | "negative"
   - question: 清晰完整的问题（中文）
   - answer: 标准答案，尽量引用或贴近原文表述
   - reason: 用 1～3 句话说明答案依据了上下文中的哪些条款/句子（包含关键原文短语）

题型定义和要求：
1）简单事实题（type = "fact"）
   - 只依赖标准片段中的“一条信息”或“一小段连续内容”即可回答。
   - 典型题目：术语定义、某个部件的检查要求、一个具体数值阈值（如电压、时间、里程）。
   - 示例风格（不要直接照抄问题）：  
     - “根据标准，在进行高压系统维修前，动力蓄电池高压输出线路系统的正负极电压必须低于多少伏才可以开始作业？”
     - “标准对动力蓄电池冷却液液面高度提出了什么具体要求？”

2）多跳推理题（type = "multi-hop"）
   - 必须要综合两条及以上信息才能得出答案，不能是简单从一条句子里直接抄答案。
   - 组合方式可以是：
     - 条件 + 附加条件（例如持证要求 + 人数要求 + 场地要求）
     - 不同章节之间的信息（例如第 4 章的一般要求 + 第 6 章的具体项目）
     - 表格中的多个条目 + 正文条款
   - 示例风格：  
     - “一名维修人员计划独自在普通维修工位上进行新能源汽车高压系统维修，他持有低压电工特种作业操作证，但场地没有设置警示隔离区。根据该片段，他是否符合标准要求？请说明理由。”
     - “在对纯电动汽车进行周期维护时，动力蓄电池系统的检查内容中，哪些项目需要结合汽车维修技术信息中的规定才能完成？”

3）否定题（type = "negative"）
   - 问的是“标准片段中【有没有/是否明确】规定某个非常具体的做法/工具/品牌/数值”等。
   - 正确答案往往是：“在该片段中未提及相关规定。”
   - 这类题用来测试模型是否会幻觉：  
     - 问题要设计得“很像应该有标准规定”，但实际片段里确实没有。
   - 示例风格：  
     - “该片段是否规定了在新能源汽车高压电气火灾时，必须优先使用某一种具体类型的灭火器（如干粉或二氧化碳）？”
   - 回答时：
     - 如果片段中确实没有相关规定，answer 必须是类似“在该片段中未提及相关规定”。
     - reason 中说明你已经查阅了相关段落，但没有找到。

题目分布要求：
- 在本次输出中，必须同时包含三种题型：fact、multi-hop、negative。
- 如果上下文片段主要是“术语和定义（第 3 章）”，建议多出 fact 题和少量多跳题，不要把所有题都做成否定题。
- 如果上下文片段包含“安全要求（第 5 章）”或“维护作业项目及要求（第 6 章）”，优先围绕安全操作、作业流程、检查要点来设计题目。
- 避免纯目录检索类问题（例如“某章节在第几页”），除非它和维修人员实际查阅标准有直接关系。

答案规范：
- answer 要尽量是可以从上下文中“逐字或近似逐字”找到支撑的内容，不要自己发挥新增技术要求。
- 对于多跳题，answer 中要把多条依据合并完整，而不是只说部分条件。
- 对于否定题，如果片段中确有相关规定，则不能回答“未提及”，而应给出正确规定并在 reason 中说明具体依据。

输出格式要求（非常重要）：
- 只能输出一个**纯 JSON 数组字符串**，不要使用 ```json 代码块，也不要添加任何解释性文字。
- 示例格式：
[
  {
    "type": "fact",
    "question": "...",
    "answer": "...",
    "reason": "..."
  },
  {
    "type": "multi-hop",
    "question": "...",
    "answer": "...",
    "reason": "..."
  }
]
"""

USER_PROMPT_TEMPLATE = """下面是标准《新能源汽车维修维护技术要求》的一个片段，请你基于这个片段，生成{num_q}道高质量试题。

【标准片段】：
{context}

请严格按照以下要求出题：

1. 所有问题都要贴近实际业务场景：
   - 角色可以是：维修技师、检验工程师、安监人员、维保主管等。
   - 问题要围绕他们“如何按标准正确维护/检修/验收车辆”“如何判断是否符合标准要求”。

2. 题型覆盖：
   - 至少 1 道简单事实题（type = "fact"）；
   - 至少 1 道多跳推理题（type = "multi-hop"）；
   - 至少 1 道否定题（type = "negative"）；
   - 如果 {num_q} > 3，可以自由增加，但三种类型都必须出现。

3. 内容取材建议：
   - 如果片段涉及第 5 章“安全要求”：优先出“安全操作流程、人员资质、场地要求、个人防护装备”等相关题。
   - 如果片段涉及第 6 章“维护作业项目及要求”：优先出“具体检查项目、合格标准、附加作业”等题。
   - 如果片段是第 3 章“术语和定义”：可以出定义题和对比题（例如高压系统 vs 专用装置），但也要设计 1 道多跳或否定题。
   - 对于“目录、前言”类片段，尽量从“如何快速查找标准中相关条款”“标准适用范围”角度来出题，避免纯页码问答。

4. 否定题的设计：
   - 问题要显得“合理且专业”，例如是否规定某种具体工具型号、某品牌设备、某精确数值。
   - 但上下文片段里确实没有这些内容。
   - 这类问题的 answer 必须是“在该片段中未提及相关规定”这一类表述。

5. 输出 JSON 时：
   - 使用字段：type, question, answer, reason
   - 不要包含其它字段。
   - 只能输出 JSON 数组本身，不要在前后增加任何说明文字，不要使用代码块标记。

现在请根据上面的要求，输出 {num_q} 条记录组成的 JSON 数组。"""


def generate_qa_for_chunk(llm: ChatOpenAI, context: str, num_q: int = 3) -> List[QAItem]:

    user_prompt = USER_PROMPT_TEMPLATE.format(context=context, num_q=num_q)
    resp = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    content = resp.content.strip()

    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3:
            lines = lines[1:]
            if lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

    content = content.strip("`").strip()

    try:
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("LLM输出不是JSON数组")
    except Exception as e:
        print("[WARN] JSON 解析失败，原始内容：", content[:300])
        raise e

    items: List[QAItem] = []
    for item in data:
        try:
            q_type_raw = item.get("type", "").strip().lower()
            if q_type_raw not in ["fact", "multi-hop", "negative"]:
                if "fact" in q_type_raw:
                    q_type_raw = "fact"
                elif "multi" in q_type_raw:
                    q_type_raw = "multi-hop"
                elif "neg" in q_type_raw or "否定" in q_type_raw:
                    q_type_raw = "negative"
                else:
                    q_type_raw = "fact"

            qa = QAItem(
                question=item["question"].strip(),
                ground_truth=item["answer"].strip(),
                context=context,
                q_type=q_type_raw,
                source_page=-1,
            )
            items.append(qa)
        except Exception as e:
            print("[WARN] 某条记录解析失败：", item, e)
            continue

    return items


def main():
    os.makedirs("outputs", exist_ok=True)

    docs = load_pdf(PDF_PATH)
    chunks = split_documents(docs, chunk_size=800, chunk_overlap=150)

    llm = get_llm()

    all_items: List[QAItem] = []

    max_chunks = max(1, TARGET_NUM_QUESTIONS // NUM_Q_PER_CHUNK)
    selected_chunks = chunks[:max_chunks]

    print(f"[INFO] 将从前 {len(selected_chunks)} 个 chunk 生成测试样本。")

    for idx, doc in enumerate(tqdm(selected_chunks, desc="Generating QAs")):
        context = doc.page_content
        page = doc.metadata.get("page", -1)

        try:
            qa_items = generate_qa_for_chunk(llm, context=context, num_q=NUM_Q_PER_CHUNK)
            for qa in qa_items:
                qa.source_page = page
            all_items.extend(qa_items)
        except Exception as e:
            print(f"[ERROR] 在第 {idx} 个 chunk 生成问答失败：", e)
            continue

    if len(all_items) > TARGET_NUM_QUESTIONS:
        all_items = all_items[:TARGET_NUM_QUESTIONS]

    print(f"[INFO] 最终生成 {len(all_items)} 条问答样本。")

    data_dicts = [asdict(item) for item in all_items]

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data_dicts, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 已保存 JSON 到 {OUTPUT_JSON}")

    df = pd.DataFrame(data_dicts)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已保存 CSV 到 {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
