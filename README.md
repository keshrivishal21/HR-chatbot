# HR Chatbot

An internal HR and company knowledge assistant built with Streamlit. Employees can ask questions about approved company documents, while role-based access control ensures that users only search the namespaces available to their role.

## Features

- Secure login backed by MongoDB and bcrypt password hashing.
- Role-based access to document namespaces:
  - **Admin:** HR policies, engineering documents, leadership documents, and general documents/FAQs.
  - **Manager:** HR policies, engineering documents, and general documents/FAQs.
  - **HR:** HR policies and general documents/FAQs.
  - **Engineer:** engineering documents and general documents/FAQs.
- Retrieval-augmented generation (RAG) using:
  - FAISS vector indexes for document search.
  - `sentence-transformers/all-MiniLM-L6-v2` embeddings.
  - Groq `llama-3.3-70b-versatile` for answer generation and grading.
- Self-correction workflow that grades retrieved documents and generated answers, rewrites queries when needed, and falls back when a reliable answer cannot be found.
- Streaming answers with retrieved source chunks and a self-correction log.
- Admin tools for uploading, re-indexing, and removing PDF, DOCX, and TXT documents.
- Admin user management, including creating users, changing roles, and deactivating accounts.

## Architecture

```text
Streamlit UI
    |
    +-- Authentication and RBAC (MongoDB)
    |
    +-- Document upload -> load -> chunk -> embed -> FAISS namespace
    |
    +-- User question -> retrieve -> grade documents
                              |
                              +-- rewrite and retry when not relevant
                              |
                              +-- generate answer -> grade grounding
                                                   |
                                                   +-- retry or return answer
```

The application stores vector indexes locally under `vector_store/faiss_index/` and stores users in the configured MongoDB database. Vector indexes are generated at runtime and are intentionally excluded from Git.

## Requirements

- Python 3.10 or newer
- A MongoDB deployment (local MongoDB or MongoDB Atlas)
- A Groq API key

## Installation

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell prevents script activation, activate the environment using Command Prompt instead:

```bat
.\.venv\Scripts\activate.bat
```

## Configuration

Copy `.env.example` to `.env` and set the required values:

```dotenv
GROQ_API_KEY=your-groq-api-key
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>/<database>
MONGODB_DB_NAME=hr_chatbot
```

Do not commit `.env` or expose API keys and database credentials.

## Seed users

After configuring MongoDB, create the initial user records:

```powershell
python seed_users.py
```

The seed script is idempotent and skips users that already exist. It currently defines these development accounts:

| Username | Password | Role |
| --- | --- | --- |
| `admin` | `admin123` | Admin |
| `manager1` | `manager123` | Manager |
| `hr1` | `hr123` | HR |
| `engineer1` | `engineer123` | Engineer |

Change or deactivate these credentials before using the application outside a development environment.

## Run the application

Start Streamlit from the project root:

```powershell
streamlit run app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`, and sign in with an active user.

## Using the application

1. Sign in with a MongoDB user account.
2. Use the **Chat** tab to ask questions about indexed company documents.
3. Admin users can use **Upload Documents** to add PDF, DOCX, or TXT files to a namespace.
4. Admin users can use **Manage Documents** to view indexed sources and remove them.
5. Admin users can use **Manage Users** to add users, update roles, or deactivate accounts.

Uploaded files are chunked and embedded into the selected namespace. Uploading a file with the same source name replaces its existing chunks in that namespace.

## Project structure

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit interface, authentication flow, chat, upload, and admin screens |
| `auth/` | MongoDB user storage and role-based access control |
| `ingestion/` | File loading, chunking, and embedding/index updates |
| `retrieval/` | Namespace-aware FAISS retrieval |
| `graph/` | LangGraph state, retrieval/generation nodes, grading, retries, and streaming |
| `utils/` | Environment configuration and shared helpers |
| `seed_users.py` | Creates initial development users |

## Notes

- General documents and general FAQs are included in every role's search scope.
- Role-specific namespaces are filtered before retrieval; the language model receives only the retrieved context.
- The application requires both `GROQ_API_KEY` and MongoDB settings at runtime.
- FAISS indexes and other generated files are ignored by Git. Back up `vector_store/` separately if indexes should be preserved between environments.

## License

See [LICENSE](LICENSE).
