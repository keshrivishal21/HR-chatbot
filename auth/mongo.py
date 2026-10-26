from datetime import datetime, timezone

import bcrypt
import streamlit as st
from pymongo import MongoClient
from pymongo.collection import Collection

from utils.helpers import get_mongodb_db_name, get_mongodb_uri, utc_now


@st.cache_resource
def get_client() -> MongoClient:
    client = MongoClient(get_mongodb_uri())
    ensure_indexes(client)
    return client


def get_users_collection() -> Collection:
    client = get_client()
    db = client[get_mongodb_db_name()]
    return db["users"]


def ensure_indexes(client: MongoClient | None = None) -> None:
    if client is None:
        client = get_client()
    db = client[get_mongodb_db_name()]
    users = db["users"]
    users.create_index("username", unique=True)
    users.create_index("role")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def authenticate(username: str, password: str) -> dict | None:
    users = get_users_collection()
    user = users.find_one({"username": username})
    if not user:
        return None
    if not user.get("is_active", False):
        return None
    if not verify_password(password, user["password_hash"]):
        return None

    now = utc_now()
    users.update_one({"username": username}, {"$set": {"last_login": now}})

    return {
        "username": user["username"],
        "role": user["role"],
        "email": user["email"],
    }


def get_all_active_users() -> list[dict]:
    users = get_users_collection()
    activeUsers = users.find({"is_active": True}, {"password_hash": 0}).sort("username", 1)
    return list(activeUsers)


def create_user(username: str, password: str, role: str, email: str) -> tuple[bool, str]:
    users = get_users_collection()
    if users.find_one({"username": username}):
        return False, "Username already exists"

    now = utc_now()
    users.insert_one(
        {
            "username": username,
            "password_hash": hash_password(password),
            "role": role,
            "email": email,
            "is_active": True,
            "created_at": now,
            "last_login": None,
        }
    )
    return True, "User created successfully"


def update_user_role(username: str, new_role: str) -> tuple[bool, str]:
    users = get_users_collection()
    result = users.update_one({"username": username, "is_active": True}, {"$set": {"role": new_role}})
    if result.matched_count == 0:
        return False, "User not found or inactive"
    return True, "Role updated successfully"


def deactivate_user(username: str, current_admin_username: str) -> tuple[bool, str]:
    if username == current_admin_username:
        return False, "You cannot deactivate your own account"

    users = get_users_collection()
    result = users.update_one(
        {"username": username, "is_active": True},
        {"$set": {"is_active": False}},
    )
    if result.matched_count == 0:
        return False, "User not found or already inactive"
    return True, "User deactivated successfully"
