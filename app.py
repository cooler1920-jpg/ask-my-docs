"""
Ask My Docs — a chatbot that answers only from information you give it.

Run locally:   streamlit run app.py
"""

import os
import re

import streamlit as st

import loaders
import rag_engine

st.set_page_config(page_title="Ask My Docs", page_icon="🤖", layout="wide")


# --------------------------------------------------------------------------
# Keys — from Streamlit secrets, then environment, then the sidebar box
# --------------------------------------------------------------------------

def stored_secret(name):
    """
    Read a key from Streamlit secrets, then from the environment.

    The template file ships with PASTE_YOUR_..._HERE placeholders. Treat those
    as missing, otherwise the app claims a key is loaded and then fails to
    connect with a confusing error.
    """
    value = ""

    try:
        if name in st.secrets:
            value = str(st.secrets[name])
    except Exception:
        pass

    value = (value or os.environ.get(name, "")).strip()

    if not value or "PASTE" in value.upper():
        return ""

    return value


SECRETS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".streamlit", "secrets.toml"
)


def save_secrets(google, pinecone, owner_password, admin_id="6002604486"):
    """
    Write the keys to .streamlit/secrets.toml so they survive a restart and
    work on other devices. This file is in .gitignore, so it never leaves
    this computer.
    """
    def clean(value):
        # TOML strings cannot contain raw quotes or backslashes.
        return value.strip().replace('"', "").replace("\\", "")

    lines = [
        "# Written by Ask My Docs. Keep this file private - never share it.",
        "",
        f'GOOGLE_API_KEY = "{clean(google)}"',
        f'PINECONE_API_KEY = "{clean(pinecone)}"',
        "",
        "# Owner login for the simple chatbot (chatbot.py)",
        f'ADMIN_ID = "{clean(admin_id)}"',
        f'ADMIN_PASSWORD = "{clean(owner_password)}"',
        "",
    ]

    folder = os.path.dirname(SECRETS_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(SECRETS_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def safe_namespace(text):
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip().lower()).strip("-")
    return cleaned or "default"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Setup")

    gemini_key = stored_secret("GOOGLE_API_KEY")
    pinecone_key = stored_secret("PINECONE_API_KEY")

    if not gemini_key:
        gemini_key = st.text_input("Google Gemini API key", type="password")
    else:
        st.success("Gemini key loaded")

    if not pinecone_key:
        pinecone_key = st.text_input("Pinecone API key", type="password")
    else:
        st.success("Pinecone key loaded")

    index_name = st.text_input("Pinecone index name", value="ask-my-docs")

    # Show this panel when the keys still need saving, or while the owner
    # password is unset - so it is never necessary to edit the file by hand.
    have_keys = bool(gemini_key and pinecone_key)
    keys_typed = have_keys and not stored_secret("GOOGLE_API_KEY")
    password_set = stored_secret("ADMIN_PASSWORD") not in ("", "change-me")

    if have_keys and (keys_typed or not password_set):
        panel_title = (
            "💾 Remember these keys on this computer"
            if keys_typed
            else "🔑 Set your owner password"
        )

        with st.expander(panel_title, expanded=True):
            if keys_typed:
                st.caption(
                    "Saves them so you never paste them again, and so the phone "
                    "version works. Stays on this computer only."
                )
            else:
                st.caption(
                    "Your keys are saved. Choose the password you will use with "
                    "ID 6002604486 to add information on the shared chatbot."
                )
            owner_password = st.text_input(
                "Choose an owner password",
                type="password",
                help="You will use this with ID 6002604486 to add information "
                     "on the simple chatbot.",
            )

            if st.button("Save on this computer", type="primary"):
                if len(owner_password.strip()) < 4:
                    st.error("Pick a password of at least 4 characters.")
                elif '"' in owner_password or "\\" in owner_password:
                    st.error(
                        'Please avoid the " and \\ characters in the password — '
                        "letters, numbers and @ # $ % are all fine."
                    )
                else:
                    try:
                        save_secrets(gemini_key, pinecone_key, owner_password)
                        st.success(
                            "Saved. Your keys and owner password are stored on "
                            "this computer."
                        )
                    except Exception as error:
                        st.error(f"Could not save: {error}")

    st.divider()
    st.caption("A knowledge base keeps one set of documents separate from another.")
    kb_label = st.text_input("Knowledge base", value="default")
    namespace = safe_namespace(kb_label)

    ready = bool(gemini_key and pinecone_key and index_name)

    if not ready:
        st.warning("Enter both API keys to begin.")

st.title("🤖 Ask My Docs")
st.caption(
    "Upload your documents, spreadsheets and photos. "
    "The bot answers **only** from what you gave it — it will not invent facts."
)

if not ready:
    st.info(
        "**First time?** Get a free Gemini key at "
        "https://aistudio.google.com/apikey and a free Pinecone key at "
        "https://app.pinecone.io — then paste both in the sidebar."
    )
    st.stop()


# --------------------------------------------------------------------------
# Connect (cached so we don't reconnect on every click)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Connecting…")
def connect(g_key, p_key, idx_name):
    return rag_engine.gemini_client(g_key), rag_engine.pinecone_index(p_key, idx_name)


try:
    client, index = connect(gemini_key, pinecone_key, index_name)
except Exception as error:
    st.error(f"Could not connect: {error}")
    st.stop()

with st.sidebar:
    st.divider()
    saved = rag_engine.stats(index, namespace)
    st.metric("Pieces of knowledge stored", saved)

    if st.button("🗑️ Clear this knowledge base", use_container_width=True):
        rag_engine.wipe(index, namespace)
        st.session_state.pop("history", None)
        st.success("Cleared.")
        st.rerun()


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

add_tab, chat_tab = st.tabs(["📥 Add knowledge", "💬 Chat"])


def saved_message(count):
    """Remember what was saved, then rerun so the sidebar counter is current."""
    word = "piece" if count == 1 else "pieces"
    st.session_state.saved_note = f"Saved {count} {word} of knowledge."
    st.rerun()


with add_tab:
    if "saved_note" in st.session_state:
        st.success(st.session_state.pop("saved_note"))

    st.subheader("Upload files")
    st.caption("PDF · Word (.docx) · Excel (.xlsx) · CSV · TXT · images (JPG, PNG, WEBP)")

    uploads = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "xlsx", "xlsm", "csv", "txt", "md", "json",
              "png", "jpg", "jpeg", "webp", "gif", "bmp"],
        accept_multiple_files=True,
    )

    notes = {}
    images = [f for f in (uploads or [])
              if os.path.splitext(f.name)[1].lower() in loaders.IMAGE_EXT]

    if images:
        st.markdown("**Explain your photos** (optional but makes answers much better)")
        for image_file in images:
            columns = st.columns([1, 3])
            with columns[0]:
                st.image(image_file, use_container_width=True)
            with columns[1]:
                notes[image_file.name] = st.text_area(
                    f"What is this? — {image_file.name}",
                    key=f"note_{image_file.name}",
                    placeholder="e.g. This is our September price list for the Guwahati branch.",
                    height=110,
                )

    if uploads and st.button("Save files to knowledge base", type="primary"):
        total = 0
        progress = st.progress(0.0)

        for position, upload in enumerate(uploads, 1):
            try:
                chunks = loaders.load(
                    client,
                    upload.name,
                    upload.getvalue(),
                    notes.get(upload.name, ""),
                )
                total += rag_engine.store_chunks(client, index, namespace, chunks)
                st.write(f"✅ {upload.name} — {len(chunks)} pieces")
            except Exception as error:
                st.write(f"❌ {upload.name} — {error}")

            progress.progress(position / len(uploads))

        saved_message(total)

    st.divider()
    st.subheader("Or type / paste information")

    pasted = st.text_area(
        "Text",
        height=180,
        placeholder="Paste FAQs, company details, policies, notes…",
    )
    pasted_label = st.text_input("Give it a name", value="Typed notes")

    if st.button("Save text to knowledge base") and pasted.strip():
        chunks = [
            {"text": piece, "source": pasted_label or "Typed notes"}
            for piece in loaders.chunk_text(pasted)
        ]
        count = rag_engine.store_chunks(client, index, namespace, chunks)
        saved_message(count)


with chat_tab:
    if "history" not in st.session_state:
        st.session_state.history = []

    if saved == 0:
        st.info("Nothing stored yet. Add some files or text in the first tab.")

    for role, message in st.session_state.history:
        with st.chat_message(role):
            st.markdown(message)

    question = st.chat_input("Ask a question about your documents…")

    if question:
        st.session_state.history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your documents…"):
                try:
                    hits = rag_engine.search(client, index, namespace, question)
                    reply, sources = rag_engine.answer(
                        client, question, hits, st.session_state.history[:-1]
                    )
                except Exception as error:
                    reply, sources, hits = f"Something went wrong: {error}", [], []

            st.markdown(reply)

            if sources:
                st.caption("Sources: " + ", ".join(sources))
                with st.expander("See the exact text this answer came from"):
                    for hit in hits:
                        st.markdown(f"**{hit['source']}** · match {hit['score']:.2f}")
                        st.text(hit["text"][:1200])
                        st.divider()

        st.session_state.history.append(("assistant", reply))
