from langchain_core.documents import Document

from ingestion.embedder import _load_vectorstore
from utils.helpers import RETRIEVAL_K, RETRIEVAL_SCORE_THRESHOLD


def retrieve(
    question: str,
    namespaces: list[str],
    k: int = RETRIEVAL_K,
    threshold: float = RETRIEVAL_SCORE_THRESHOLD,
) -> list[Document]:
    scored: list[tuple[Document, float]] = []

    for namespace in namespaces:
        vectorstore = _load_vectorstore(namespace)
        if vectorstore is None:
            continue

        for doc, score in vectorstore.similarity_search_with_score(question, k=k):
            scored.append((doc, score))

    scored.sort(key=lambda x: x[1])

    filtered = [(doc, score) for doc, score in scored if score <= threshold]

    best_by_content: dict[str, tuple[Document, float]] = {}
    for doc, score in filtered:
        content_key = doc.page_content.strip()
        if content_key not in best_by_content or score < best_by_content[content_key][1]:
            best_by_content[content_key] = (doc, score)

    deduped = sorted(best_by_content.values(), key=lambda x: x[1])
    return [doc for doc, _ in deduped[:k]]
