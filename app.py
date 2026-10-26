import streamlit as st

from auth.mongo import (
    authenticate,
    create_user,
    deactivate_user,
    get_all_active_users,
    update_user_role,
)
from auth.rbac import ALL_NAMESPACES, ALL_ROLES, can_manage_users, can_upload, get_accessible_namespaces
from graph.workflow import run_graph_with_stream
from ingestion.chunker import chunk_documents
from ingestion.embedder import delete_document, list_sources, upsert_documents
from ingestion.loader import load_file
from utils.helpers import format_role_badge

st.set_page_config(
    page_title="Internal Company Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Light styling polish (no behavior changes — purely cosmetic)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        div[data-testid="stForm"] {
            border: 1px solid rgba(49, 51, 63, 0.15);
            border-radius: 10px;
            padding: 1.5rem 1.5rem 0.5rem 1.5rem;
        }
        .source-chip {
            display: inline-block;
            background: rgba(49, 51, 63, 0.06);
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 0.8rem;
            margin-right: 6px;
        }
        .row-divider {margin-top: 0.25rem; margin-bottom: 0.25rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    defaults = {
        "authenticated": False,
        "username": "",
        "role": "",
        "email": "",
        "messages": [],
        "confirm_delete_source": None,
        "confirm_deactivate_user": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def logout() -> None:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.session_state.email = ""
    st.session_state.messages = []


def render_login() -> None:
    left, mid, right = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 💬 Internal Company Chatbot")
        st.caption("Sign in with your company credentials to continue.")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="e.g. jane.doe")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Log in", use_container_width=True)

        if submitted:
            if not username or not password:
                st.error("Please enter both username and password.")
                return

            with st.spinner("Signing in..."):
                user = authenticate(username, password)

            if user:
                st.session_state.authenticated = True
                st.session_state.username = user["username"]
                st.session_state.role = user["role"]
                st.session_state.email = user["email"]
                st.session_state.messages = []
                st.rerun()
            else:
                st.error("Invalid credentials or inactive account.")


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 💬 Company Chatbot")
        st.divider()

        st.markdown(f"**{st.session_state.username}**")
        st.markdown(format_role_badge(st.session_state.role), unsafe_allow_html=True)
        st.caption(st.session_state.email)

        st.divider()

        if st.button("🚪 Logout", use_container_width=True):
            logout()
            st.rerun()


def render_chat_tab() -> None:
    st.title("Company Knowledge Assistant")
    st.caption("Ask questions about internal company documents.")

    messages_container = st.container(height=500)
    with messages_container:
        if not st.session_state.messages:
            st.info("👋 Ask a question to get started.")
        for message in st.session_state.messages:
            avatar = "🧑‍💼" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

                if message["role"] == "assistant" and "metadata" in message:
                    meta = message["metadata"]

                    with st.expander("📄 Source Documents", expanded=False):
                        docs = meta.get("documents", [])
                        if not docs:
                            st.info("No source documents were retrieved.")
                        for i, doc in enumerate(docs, 1):
                            source = doc.metadata.get("source", "unknown")
                            namespace = doc.metadata.get("namespace", "unknown")
                            st.markdown(
                                f"**Chunk {i}** "
                                f"<span class='source-chip'>{source}</span>"
                                f"<span class='source-chip'>{namespace}</span>",
                                unsafe_allow_html=True,
                            )
                            st.text(doc.page_content[:1000])
                            if i < len(docs):
                                st.markdown("<hr class='row-divider'>", unsafe_allow_html=True)

                    with st.expander("🔁 Self-Correction Log", expanded=False):
                        col1, col2 = st.columns(2)
                        col1.metric("Retry count", meta.get("retry_count", 0))
                        col2.metric("Final relevance score", meta.get("final_relevance_score", "unknown"))
                        log = meta.get("correction_log", [])
                        if log:
                            for entry in log:
                                st.json(entry)
                        else:
                            st.info("No correction steps recorded.")

    if prompt := st.chat_input("Ask a question about company documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant", avatar="🤖"):
            status_box = st.status("Searching documents...", expanded=False)
            answer_placeholder = st.empty()
            accumulated = ""
            result = {}

            namespaces = get_accessible_namespaces(st.session_state.role)
            for event_type, payload in run_graph_with_stream(
                prompt, st.session_state.role, namespaces
            ):
                if event_type == "status":
                    status_box.update(label=payload)
                elif event_type == "token":
                    accumulated += payload
                    answer_placeholder.markdown(accumulated + "▌")
                elif event_type == "done":
                    result = payload

            status_box.update(label="Done", state="complete", expanded=False)
            answer_placeholder.markdown(accumulated)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": accumulated,
                "metadata": {
                    "documents": result.get("documents", []),
                    "correction_log": result.get("correction_log", []),
                    "retry_count": result.get("retry_count", 0),
                    "final_relevance_score": result.get("final_relevance_score", "unknown"),
                },
            }
        )
        st.rerun()


def render_upload_tab() -> None:
    st.title("Upload Documents")
    st.caption("Index PDF, DOCX, or TXT files into a document namespace.")
    st.divider()

    uploaded_files = st.file_uploader(
        "Select documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Accepted formats: PDF, DOCX, TXT. You can select multiple files at once.",
    )
    namespace = st.selectbox(
        "Target namespace",
        ALL_NAMESPACES,
        key="upload_namespace",
        help="Documents are indexed per-namespace; only roles with access to this namespace will see it in chat.",
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected.")

    st.divider()

    if st.button("⬆️ Upload & Index", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please select at least one file.")
        else:
            total_chunks = 0
            failures = []
            progress = st.progress(0, text="Starting...")
            for idx, uploaded in enumerate(uploaded_files, start=1):
                progress.progress(
                    idx / len(uploaded_files), text=f"Processing {uploaded.name} ({idx}/{len(uploaded_files)})..."
                )
                try:
                    docs = load_file(uploaded.read(), uploaded.name)
                    chunks = chunk_documents(docs, uploaded.name, namespace)
                    total_chunks += upsert_documents(namespace, chunks)
                except Exception as exc:
                    failures.append((uploaded.name, str(exc)))
            progress.empty()

            if total_chunks:
                st.success(f"✅ Indexed {total_chunks} chunk(s) into '{namespace}'.")
            for name, err in failures:
                st.error(f"Failed to process {name}: {err}")


def render_manage_documents_tab() -> None:
    st.title("Manage Documents")
    st.caption("View and remove indexed documents by namespace.")
    st.divider()

    namespace = st.selectbox("Namespace", ALL_NAMESPACES, key="manage_namespace")
    sources = list_sources(namespace)

    st.divider()

    if not sources:
        st.info("No documents indexed in this namespace.")
        return

    search = st.text_input("🔍 Filter by filename", placeholder="Start typing to filter...")
    filtered_sources = [s for s in sources if search.lower() in s.lower()] if search else sources

    st.markdown(f"**{len(filtered_sources)} of {len(sources)} document(s) in `{namespace}`**")
    st.divider()

    if not filtered_sources:
        st.info("No documents match your filter.")
        return

    header_cols = st.columns([4, 1])
    header_cols[0].markdown("**Document**")
    header_cols[1].markdown("**Action**")

    for source in filtered_sources:
        cols = st.columns([4, 1])
        with cols[0]:
            st.text(source)
        with cols[1]:
            pending = st.session_state.confirm_delete_source == source
            if not pending:
                if st.button("Remove", key=f"remove_{namespace}_{source}", use_container_width=True):
                    st.session_state.confirm_delete_source = source
                    st.rerun()
            else:
                if st.button("Confirm ✅", key=f"confirm_{namespace}_{source}", use_container_width=True, type="primary"):
                    deleted = delete_document(namespace, source)
                    st.session_state.confirm_delete_source = None
                    st.success(f"Removed {deleted} chunk(s) from '{source}'.")
                    st.rerun()
        if pending:
            cancel_cols = st.columns([4, 1])
            with cancel_cols[1]:
                if st.button("Cancel", key=f"cancel_{namespace}_{source}", use_container_width=True):
                    st.session_state.confirm_delete_source = None
                    st.rerun()


def render_user_management_tab() -> None:
    st.title("Manage Users")
    st.caption("Add users, change roles, and deactivate accounts.")
    st.divider()

    with st.form("add_user_form"):
        st.markdown("**Add New User**")
        new_username = st.text_input("Username", key="new_username")
        new_password = st.text_input("Password", type="password", key="new_password")
        new_role = st.selectbox("Role", ALL_ROLES, key="new_role")
        new_email = st.text_input("Email", key="new_email")
        add_submitted = st.form_submit_button("Add User", use_container_width=True)

    if add_submitted:
        if not all([new_username, new_password, new_role, new_email]):
            st.warning("All fields are required.")
        else:
            success, message = create_user(new_username, new_password, new_role, new_email)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    st.divider()
    st.markdown("**Active Users**")
    users = get_all_active_users()

    if not users:
        st.info("No active users found.")
        return

    st.divider()

    header_cols = st.columns([2, 2, 2, 1])
    header_cols[0].markdown("**Username**")
    header_cols[1].markdown("**Email**")
    header_cols[2].markdown("**Role**")
    header_cols[3].markdown("**Action**")

    for user in users:
        cols = st.columns([2, 2, 2, 1])
        with cols[0]:
            st.text(user["username"])
        with cols[1]:
            st.text(user["email"])
        with cols[2]:
            role_key = f"role_{user['username']}"
            selected_role = st.selectbox(
                "Role",
                ALL_ROLES,
                index=ALL_ROLES.index(user["role"]),
                key=role_key,
                label_visibility="collapsed",
            )
            if selected_role != user["role"]:
                success, message = update_user_role(user["username"], selected_role)
                if success:
                    st.rerun()
                else:
                    st.error(message)
        with cols[3]:
            is_self = user["username"] == st.session_state.username
            pending = st.session_state.confirm_deactivate_user == user["username"]

            if is_self:
                st.button("Deactivate", key=f"deact_{user['username']}", disabled=True, use_container_width=True)
            elif not pending:
                if st.button("Deactivate", key=f"deact_{user['username']}", use_container_width=True):
                    st.session_state.confirm_deactivate_user = user["username"]
                    st.rerun()
            else:
                if st.button("Confirm ✅", key=f"deact_confirm_{user['username']}", use_container_width=True, type="primary"):
                    success, message = deactivate_user(user["username"], st.session_state.username)
                    st.session_state.confirm_deactivate_user = None
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

        if pending and not is_self:
            cancel_cols = st.columns([2, 2, 2, 1])
            with cancel_cols[3]:
                if st.button("Cancel", key=f"deact_cancel_{user['username']}", use_container_width=True):
                    st.session_state.confirm_deactivate_user = None
                    st.rerun()


def render_main_tabs() -> None:
    role = st.session_state.role

    tab_labels = ["💬 Chat"]
    tab_renderers = [render_chat_tab]

    if can_upload(role):
        tab_labels += ["📤 Upload Documents", "🗑️ Manage Documents"]
        tab_renderers += [render_upload_tab, render_manage_documents_tab]

    if can_manage_users(role):
        tab_labels += ["👥 Manage Users"]
        tab_renderers.append(render_user_management_tab)

    tabs = st.tabs(tab_labels)
    for tab, renderer in zip(tabs, tab_renderers):
        with tab:
            renderer()


def main() -> None:
    init_session_state()

    if not st.session_state.authenticated:
        render_login()
        return

    render_sidebar()
    render_main_tabs()


if __name__ == "__main__":
    main()