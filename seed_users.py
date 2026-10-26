"""
Seed initial users into MongoDB Atlas.
Safe to run multiple times — uses upsert on username.
"""

from datetime import datetime, timezone

import bcrypt
from dotenv import load_dotenv
from pymongo import MongoClient

from utils.helpers import get_mongodb_db_name, get_mongodb_uri

load_dotenv()

SEED_USERS = [
    {
        "username": "admin",
        "password": "admin123",
        "role": "Admin",
        "email": "admin@company.com",
    },
    {
        "username": "manager1",
        "password": "manager123",
        "role": "Manager",
        "email": "manager@company.com",
    },
    {
        "username": "hr1",
        "password": "hr123",
        "role": "HR",
        "email": "hr@company.com",
    },
    {
        "username": "engineer1",
        "password": "engineer123",
        "role": "Engineer",
        "email": "engineer@company.com",
    },
]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def seed_users() -> None:
    client = MongoClient(get_mongodb_uri())
    db = client[get_mongodb_db_name()]
    users = db["users"]

    users.create_index("username", unique=True)
    users.create_index("role")

    now = datetime.now(timezone.utc)

    for user in SEED_USERS:
        password_hash = hash_password(user["password"])
        result = users.update_one(
            {"username": user["username"]},
            {
                "$setOnInsert": {
                    "username": user["username"],
                    "password_hash": password_hash,
                    "role": user["role"],
                    "email": user["email"],
                    "is_active": True,
                    "created_at": now,
                    "last_login": None,
                }
            },
            upsert=True,
        )
        if result.upserted_id:
            print(f"Created user: {user['username']} ({user['role']})")
        else:
            print(f"User already exists: {user['username']} (skipped)")

    print("Seed complete.")


if __name__ == "__main__":
    seed_users()
