import streamlit as st
from collections.abc import Iterator
from typing import Any

from langgraph.graph import END, StateGraph

from graph.nodes import (
    FALLBACK_MESSAGE,
    finalize_node,
    generate_node,
    grade_documents_node,
    grade_generation_node,
    retrieve_node,
    rewrite_query_node,
    stream_generation,
)
from graph.state import GraphState


def _route_after_grade_documents(state: GraphState) -> str:
    if state.get("relevance_grade") == "relevant":
        return "generate"
    return "rewrite_query"


def _route_after_grade_generation(state: GraphState) -> str:
    if state.get("grounding_grade") == "grounded":
        return "finalize"
    return "rewrite_query"


def _route_after_rewrite(state: GraphState) -> str:
    if state.get("force_fallback"):
        return "finalize"
    return "retrieve"


def _build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade_generation", grade_generation_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        _route_after_grade_documents,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    workflow.add_edge("generate", "grade_generation")
    workflow.add_conditional_edges(
        "grade_generation",
        _route_after_grade_generation,
        {"finalize": "finalize", "rewrite_query": "rewrite_query"},
    )
    workflow.add_conditional_edges(
        "rewrite_query",
        _route_after_rewrite,
        {"retrieve": "retrieve", "finalize": "finalize"},
    )
    workflow.add_edge("finalize", END)

    return workflow.compile()


@st.cache_resource
def get_compiled_graph():
    return _build_graph()


def _build_initial_state(question: str, user_role: str, accessible_namespaces: list[str]) -> GraphState:
    return {
        "question": question,
        "rewritten_question": "",
        "user_role": user_role,
        "accessible_namespaces": accessible_namespaces,
        "documents": [],
        "generation": "",
        "relevance_grade": "",
        "grounding_grade": "",
        "retry_count": 0,
        "final_answer": "",
        "correction_log": [],
        "force_fallback": False,
    }


def _build_result(state: GraphState) -> dict:
    relevance_scores = [
        entry.get("relevance_grade")
        for entry in state.get("correction_log", [])
        if entry.get("step") == "grade_documents"
    ]
    final_relevance = relevance_scores[-1] if relevance_scores else "unknown"

    return {
        "final_answer": state.get("final_answer", ""),
        "documents": state.get("documents", []),
        "correction_log": state.get("correction_log", []),
        "retry_count": state.get("retry_count", 0),
        "final_relevance_score": final_relevance,
    }


def _merge_state(state: GraphState, update: dict) -> GraphState:
    return {**state, **update}


def run_graph_with_stream(
    question: str, user_role: str, accessible_namespaces: list[str]
) -> Iterator[tuple[str, Any]]:
    """Yield ('status', label), ('token', chunk), then ('done', result dict)."""
    state = _build_initial_state(question, user_role, accessible_namespaces)

    while True:
        yield ("status", "Searching documents...")
        state = _merge_state(state, retrieve_node(state))

        yield ("status", "Checking document relevance...")
        state = _merge_state(state, grade_documents_node(state))

        if state.get("relevance_grade") != "relevant":
            state = _merge_state(state, rewrite_query_node(state))
            if state.get("force_fallback"):
                for char in FALLBACK_MESSAGE:
                    yield ("token", char)
                state = _merge_state(state, finalize_node(state))
                yield ("done", _build_result(state))
                return
            yield ("status", "Rewriting query and retrying...")
            continue

        yield ("status", "Generating answer...")
        token_buffer: list[str] = []
        for token in stream_generation(state):
            token_buffer.append(token)
        state = _merge_state(state, {"generation": "".join(token_buffer)})

        yield ("status", "Verifying answer...")
        state = _merge_state(state, grade_generation_node(state))

        if state.get("grounding_grade") == "grounded":
            state = _merge_state(state, finalize_node(state))
            for token in token_buffer:
                yield ("token", token)
            yield ("done", _build_result(state))
            return

        state = _merge_state(state, rewrite_query_node(state))
        if state.get("force_fallback"):
            for char in FALLBACK_MESSAGE:
                yield ("token", char)
            state = _merge_state(state, finalize_node(state))
            yield ("done", _build_result(state))
            return
        yield ("status", "Answer not grounded — retrying...")


def run_graph(question: str, user_role: str, accessible_namespaces: list[str]) -> dict:
    graph = get_compiled_graph()

    initial_state = _build_initial_state(question, user_role, accessible_namespaces)

    result = graph.invoke(initial_state)

    relevance_scores = [
        entry.get("relevance_grade")
        for entry in result.get("correction_log", [])
        if entry.get("step") == "grade_documents"
    ]
    final_relevance = relevance_scores[-1] if relevance_scores else "unknown"

    return {
        "final_answer": result.get("final_answer", ""),
        "documents": result.get("documents", []),
        "correction_log": result.get("correction_log", []),
        "retry_count": result.get("retry_count", 0),
        "final_relevance_score": final_relevance,
    }
