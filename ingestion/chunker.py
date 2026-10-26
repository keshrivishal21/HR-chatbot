import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.helpers import utc_now_iso

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_documents(
    documents: list[Document],
    source: str, #orignal file name
    namespace: str,  #who can access this document
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    uploaded_at = utc_now_iso()

    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = source
        chunk.metadata["namespace"] = namespace
        chunk.metadata["uploaded_at"] = uploaded_at
        chunk.metadata["chunk_id"] = hashlib.md5(
            f"{namespace}-{source}-{i}".encode()
        ).hexdigest()

    return chunks
