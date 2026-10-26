import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
RETRIEVAL_K = 4
RETRIEVAL_SCORE_THRESHOLD = 1.5  # L2 distance; lower = more similar
MAX_RETRIES = 2

ROLE_COLORS = {
    "Admin": "#FF4B4B",
    "Manager": "#0068C9",
    "HR": "#09AB3B",
    "Engineer": "#FFA421",
}


def get_groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is not set in environment")
    return key


def get_mongodb_uri() -> str:
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI is not set in environment")
    return uri


def get_mongodb_db_name() -> str:
    name = os.getenv("MONGODB_DB_NAME")
    if not name:
        raise ValueError("MONGODB_DB_NAME is not set in environment")
    return name


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def format_role_badge(role: str) -> str:
    color = ROLE_COLORS.get(role, "#808495")
    return f'<span style="background-color:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.85em;">{role}</span>'
