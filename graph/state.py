from typing import Literal, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    rewritten_question: str
    user_role: str
    accessible_namespaces: list[str]
    documents: list[Document]
    generation: str
    relevance_grade: Literal["", "relevant", "not_relevant"]
    grounding_grade: Literal["", "grounded", "hallucinated"]
    retry_count: int
    final_answer: str
    correction_log: list[dict]
    force_fallback: bool
