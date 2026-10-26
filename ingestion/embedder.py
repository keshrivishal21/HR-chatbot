from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from utils.helpers import EMBEDDINGS_MODEL

VECTOR_STORE_ROOT = Path(__file__).resolve().parent.parent / "vector_store" / "faiss_index"


@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)


def get_index_path(namespace: str) -> Path:
    return VECTOR_STORE_ROOT / namespace


def _load_vectorstore(namespace: str) -> FAISS | None:
    index_path = get_index_path(namespace)
    if not (index_path / "index.faiss").exists():
        return None

    embeddings = get_embeddings()
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _ids_for_source(vectorstore: FAISS, source: str) -> list[str]:
    return [
        doc_id
        for doc_id, doc in vectorstore.docstore._dict.items()
        if doc.metadata.get("source") == source
    ]


def list_sources(namespace: str) -> list[str]:
    vectorstore = _load_vectorstore(namespace)
    if vectorstore is None:
        return []

    sources = {
        doc.metadata.get("source")
        for doc in vectorstore.docstore._dict.values()
        if doc.metadata.get("source")
    }
    return sorted(sources)


def delete_document(namespace: str, source: str) -> int:
    vectorstore = _load_vectorstore(namespace)
    if vectorstore is None:
        return 0

    ids_to_delete = _ids_for_source(vectorstore, source)
    if not ids_to_delete:
        return 0

    vectorstore.delete(ids=ids_to_delete)
    vectorstore.save_local(str(get_index_path(namespace)))
    return len(ids_to_delete)


def upsert_documents(namespace: str, documents: list[Document]) -> int:
    if not documents:
        return 0

    embeddings = get_embeddings()
    index_path = get_index_path(namespace)
    index_path.mkdir(parents=True, exist_ok=True)

    if (index_path / "index.faiss").exists():
        vectorstore = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        vectorstore = None

    sources = {doc.metadata.get("source") for doc in documents if doc.metadata.get("source")}
    if vectorstore is not None:
        for source in sources:
            ids_to_delete = _ids_for_source(vectorstore, source)
            if ids_to_delete:
                vectorstore.delete(ids=ids_to_delete)

    if vectorstore is None:
        vectorstore = FAISS.from_documents(documents, embeddings)
    else:
        vectorstore.add_documents(documents)

    vectorstore.save_local(str(index_path))
    return len(documents)
