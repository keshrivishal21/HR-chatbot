from typing import Iterator, Literal

from pydantic import BaseModel
import streamlit as st
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from graph.state import GraphState
from retrieval.retriever import retrieve
from utils.helpers import GROQ_MODEL, MAX_RETRIES, get_groq_api_key

FALLBACK_MESSAGE = (
    "I was unable to find a reliable answer in the available company documents. "
    "Please contact your administrator for assistance."
)


class RelevanceGrade(BaseModel):
    grade: Literal["relevant", "not_relevant"]


class GroundingGrade(BaseModel):
    grade: Literal["grounded", "hallucinated"]


@st.cache_resource
def _get_llm() -> ChatGroq:
    return ChatGroq(model=GROQ_MODEL, api_key=get_groq_api_key(), temperature=0)


def _format_docs(documents: list[Document]) -> str:
    if not documents:
        return "No documents retrieved."
    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        namespace = doc.metadata.get("namespace", "unknown")
        parts.append(
            f"[Document {i} | source: {source} | namespace: {namespace}]\n{doc.page_content}"
        )
    return "\n\n".join(parts)


def retrieve_node(state: GraphState) -> dict:
    query = state.get("rewritten_question") or state["question"]
    documents = retrieve(query, state["accessible_namespaces"])
    return {"documents": documents}


def grade_documents_node(state: GraphState) -> dict:
    llm = _get_llm()
    question = state.get("rewritten_question") or state["question"]
    docs_text = _format_docs(state["documents"])

    prompt = f"""You are a document relevance grader. Assess whether the retrieved documents are relevant to the user question.

User question: {question}

Retrieved documents:
{docs_text}

Respond with a structured grade: "relevant" if the documents contain information useful for answering the question, or "not_relevant" if they do not."""

    grader = llm.with_structured_output(RelevanceGrade)
    try:
        grade = grader.invoke(prompt).grade
    except Exception:
        grade = "not_relevant"

    log_entry = {
        "attempt": state["retry_count"] + 1,
        "step": "grade_documents",
        "relevance_grade": grade,
        "query_used": question,
    }

    return {
        "relevance_grade": grade,
        "correction_log": state.get("correction_log", []) + [log_entry],
    }


def _generation_messages(state: GraphState) -> list[dict]:
    question = state.get("rewritten_question") or state["question"]
    context = _format_docs(state["documents"])
    role = state["user_role"]

    system_prompt = f"""You are an internal company assistant helping a {role} employee.
Answer the question using ONLY the provided context documents.
If the context does not contain enough information to answer, say you cannot find the information in the available documents.
Never make up or infer information that is not explicitly supported by the context."""

    user_prompt = f"""Context documents:
{context}

Question: {question}

Provide a clear, concise answer based solely on the context above."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_node(state: GraphState) -> dict:
    llm = _get_llm()
    response = llm.invoke(_generation_messages(state))
    return {"generation": response.content}


def stream_generation(state: GraphState) -> Iterator[str]:
    llm = _get_llm()
    for chunk in llm.stream(_generation_messages(state)):
        if chunk.content:
            yield chunk.content


def grade_generation_node(state: GraphState) -> dict:
    llm = _get_llm()
    question = state.get("rewritten_question") or state["question"]
    docs_text = _format_docs(state["documents"])
    generation = state["generation"]

    prompt = f"""You are a hallucination grader. Determine whether the generated answer is grounded in the retrieved documents.

User question: {question}

Retrieved documents:
{docs_text}

Generated answer:
{generation}

Respond with a structured grade: "grounded" if every claim in the answer is supported by the documents, or "hallucinated" if the answer contains unsupported or fabricated information."""

    grader = llm.with_structured_output(GroundingGrade)
    try:
        grade = grader.invoke(prompt).grade
    except Exception:
        grade = "hallucinated"

    log_entry = {
        "attempt": state["retry_count"] + 1,
        "step": "grade_generation",
        "grounding_grade": grade,
    }

    return {
        "grounding_grade": grade,
        "correction_log": state.get("correction_log", []) + [log_entry],
    }


def rewrite_query_node(state: GraphState) -> dict:
    llm = _get_llm()
    question = state["question"]
    retry_count = state["retry_count"] + 1

    if retry_count > MAX_RETRIES:
        return {
            "retry_count": retry_count,
            "force_fallback": True,
            "correction_log": state.get("correction_log", [])
            + [
                {
                    "attempt": retry_count,
                    "step": "rewrite_query",
                    "action": "max_retries_exceeded",
                }
            ],
        }

    prompt = f"""You are a query rewriter for a company document search system.
Rewrite the following question to improve document retrieval. Keep the original intent but use different wording or add relevant keywords.

Original question: {question}

Return only the rewritten question, nothing else."""

    response = llm.invoke(prompt)
    rewritten = response.content.strip()

    log_entry = {
        "attempt": retry_count,
        "step": "rewrite_query",
        "original_question": question,
        "rewritten_question": rewritten,
    }

    return {
        "rewritten_question": rewritten,
        "retry_count": retry_count,
        "correction_log": state.get("correction_log", []) + [log_entry],
    }


def finalize_node(state: GraphState) -> dict:
    if state.get("force_fallback"):
        final_answer = FALLBACK_MESSAGE
    elif state.get("grounding_grade") == "grounded" and state.get("generation"):
        final_answer = state["generation"]
    elif state.get("relevance_grade") == "not_relevant" and state["retry_count"] >= MAX_RETRIES:
        final_answer = FALLBACK_MESSAGE
    else:
        final_answer = state.get("generation") or FALLBACK_MESSAGE

    return {"final_answer": final_answer}
