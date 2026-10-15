import os
import uuid
import gradio as gr

# LangChain 관련 모듈 Import
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langsmith import Client


def build_db(pdf_path, openai_key):
    if not openai_key:
        return None, "⚠️ OpenAI API 키를 먼저 입력한 뒤 PDF를 다시 올려주세요."
    if pdf_path is None:
        return None, ""

    pages = PyPDFLoader(pdf_path).load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=25,
        length_function=len, is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-large", openai_api_key=openai_key,
    )

    db = Chroma.from_documents(
        texts, embeddings_model, collection_name=f"pdf_{uuid.uuid4().hex}"
    )
    return db, f"✅ 처리 완료:{len(pages)}페이지 →{len(texts)}개 청크"

def answer(question, openai_key, db):
    if db is None:
        raise gr.Error("먼저 API 키를 입력하고 PDF를 업로드하세요.")
    if not question:
        raise gr.Error("질문을 입력하세요.")

    llm = ChatOpenAI(temperature=0, openai_api_key=openai_key)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(), llm=llm
    )
#     prompt = ChatPromptTemplate.from_messages([
#     ("system", "당신은 설명서를 바탕으로 답하는 안내 도우미입니다.\n\n" 
#      "아래 [자료]에 있는 내용만 근거로 한국어로 답하세요\n"
#      "자료에 없으면 '설명서에서 확인되지 않습니다.'라고만 답하세요.\n"
#      "자료에 '불가능하다', '받지 않는다'처럼 없음이나 금지로 적힌 내용도 확인된 내용이므로 그대로 답하세요.\n"
#      "답변은 자료에서 확인될 때만 마지막 줄에 출처(페이지)를 적으세요.\n\n"
#      "[자료]\n{context}"),
#     ("human", "{question}"),])

    client = Client()
    prompt = client.pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)
    generate_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_key)

    def format_docs(docs):
        return "\n\n".join(
            f"[출처: {doc.metadata.get('page', '?')}쪽]\n{doc.page_content}" for doc in docs
        )

    rag_chain = (
        {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
        | prompt
        | generate_llm
        | StrOutputParser()
    )
    yield "🔎 설명서에서 찾는 중..."          # 상태 메시지를 먼저 화면에 내보냄
    
    text = ""
    for chunk in rag_chain.stream(question):
        text += chunk
        yield text

with gr.Blocks(title="ChatPDF") as demo:
    gr.Markdown("# ChatPDF\n---")

    openai_key = gr.Textbox(label="OPEN_AI_API_KEY", type="password")
    pdf_file = gr.File(label="PDF 파일을 올려주세요!", file_types=[".pdf"], type="filepath")
    status = gr.Markdown()
    db_state = gr.State()

    gr.Markdown("---\n## PDF에게 질문해보세요!!")
    question = gr.Textbox(label="질문을 입력하세요")
    ask_btn = gr.Button("질문하기", variant="primary")
    output = gr.Markdown()

    gr.Markdown("[☕ Buy me a coffee](https://www.buymeacoffee.com/{계정 ID})")

    pdf_file.upload(build_db, inputs=[pdf_file, openai_key], outputs=[db_state, status])
    ask_btn.click(answer, inputs=[question, openai_key, db_state], outputs=output)
    question.submit(answer, inputs=[question, openai_key, db_state], outputs=output)

# =====================================================================
# [추가] Baseline 측정 — evaluation_questions.csv를 불러와 기존 answer()로 답변 후 채점
#   - 기존 build_db / answer 함수를 그대로 호출하므로 현재 파이프라인의 성능이 그대로 측정됩니다.
#   - 위에서 업로드한 PDF로 만든 DB(db_state)를 그대로 사용합니다.
# =====================================================================
import tempfile
import pandas as pd
from pydantic import BaseModel, Field
 
 
class EvalGrade(BaseModel):
    correct: bool = Field(description="모델 답변이 정답과 의미상 일치하면 True, 아니면 False")
    reason: str = Field(description="판정 근거 (한 문장)")
 
 
EVAL_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 RAG 시스템의 답변을 채점하는 평가자입니다.\n"
     "[질문], [정답], [모델 답변]을 보고 모델 답변이 정답과 의미상 일치하는지 판정하세요.\n"
     "- 표현이나 언어(한국어/영어)가 달라도 핵심 내용이 같으면 정답입니다.\n"
     "- 정답이 '확인할 수 없다/알 수 없다'인 경우, 모델이 모른다(I don't know 등) 또는 "
     "문서에서 확인되지 않는다고 답하면 정답이고, 구체적인 값을 추측해 단정하면 오답입니다.\n"
     "- 정답과 다른 사실을 단정하거나 핵심 내용이 빠지면 오답입니다."),
    ("human", "[질문]\n{question}\n\n[정답]\n{reference}\n\n[모델 답변]\n{prediction}"),
])
 
 
def eval_question_type(reason):
    """reason 컬럼 앞부분으로 질문 유형 분류 (문서 기반 / 표현 변경 / 문서 외)."""
    for t in ("문서 기반", "표현 변경", "문서 외"):
        if str(reason).strip().startswith(t):
            return t
    return "기타"
 
 
def run_baseline(csv_path, openai_key, db, progress=gr.Progress()):
    if not openai_key:
        raise gr.Error("OpenAI API 키를 먼저 입력하세요.")
    if csv_path is None:
        raise gr.Error("평가 질문 CSV 파일을 올려주세요.")
    if db is None:
        raise gr.Error("먼저 위에서 PDF를 업로드해 DB를 만들어주세요.")
 
    # 1) CSV 불러오기 (BOM 대응, 'anwser' 오타 컬럼 대응)
    df = pd.read_csv(csv_path, encoding="utf-8-sig").dropna(subset=["question"])
    answer_col = "answer" if "answer" in df.columns else "anwser"
    df = df.reset_index(drop=True)
 
    # 2) 질문별로 기존 answer() 실행 → 채점
    judge = ChatOpenAI(
        model="gpt-4o-mini", temperature=0, openai_api_key=openai_key
    ).with_structured_output(EvalGrade)
    grader = EVAL_GRADE_PROMPT | judge
 
    rows = []
    n = len(df)
    for i, r in df.iterrows():
        progress((i, n), desc=f"질문 {i + 1}/{n} 평가 중...")
        q = str(r["question"])
        ref = str(r[answer_col])
        try:
            prediction = ""
            for out in answer(q, openai_key, db):   # 기존 answer() 그대로 사용 (마지막 yield가 최종 답변)
                prediction = out
        except Exception as e:
            prediction = f"[ERROR] {e}"
 
        try:
            g = grader.invoke({"question": q, "reference": ref, "prediction": prediction})
            correct, judge_reason = bool(g.correct), g.reason
        except Exception as e:
            correct, judge_reason = False, f"[채점 실패] {e}"
 
        rows.append({
            "no": i + 1,
            "type": eval_question_type(r.get("reason", "")),
            "question": q,
            "reference": ref,
            "prediction": prediction,
            "correct": correct,
            "judge_reason": judge_reason,
            "evidence_page": r.get("evidence_page", ""),
        })
 
    result = pd.DataFrame(rows)
 
    # 3) 요약 (전체 + 유형별 정확도)
    total_acc = result["correct"].mean() * 100
    by_type = result.groupby("type")["correct"].agg(["sum", "count"])
    lines = [
        "### 📊 Baseline 결과",
        f"**전체 정확도: {total_acc:.1f}% ({int(result['correct'].sum())}/{len(result)})**",
        "",
        "| 질문 유형 | 정답 수 | 문항 수 | 정확도 |",
        "|---|---|---|---|",
    ]
    for t, s in by_type.iterrows():
        lines.append(f"| {t} | {int(s['sum'])} | {int(s['count'])} | {s['sum'] / s['count'] * 100:.1f}% |")
    summary = "\n".join(lines)
 
    out_path = os.path.join(tempfile.gettempdir(), f"baseline_result_{uuid.uuid4().hex[:8]}.csv")
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    return summary, result, out_path
 
 
with demo:
    gr.Markdown("---\n## 📊 Baseline 측정")
    eval_csv = gr.File(label="평가 질문 CSV (evaluation_questions.csv)", file_types=[".csv"], type="filepath")
    eval_btn = gr.Button("Baseline 측정 시작")
    eval_summary = gr.Markdown()
    eval_table = gr.Dataframe(label="문항별 결과", wrap=True)
    eval_download = gr.File(label="결과 CSV 다운로드")
 
    eval_btn.click(
        run_baseline,
        inputs=[eval_csv, openai_key, db_state],
        outputs=[eval_summary, eval_table, eval_download],
    )

if __name__ == "__main__":
    demo.launch()