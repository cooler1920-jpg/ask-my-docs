"""
Simple public chatbot.

Visitors see one thing: a chat box.
Only the owner (signed in) can add information.

Run:  .venv\\Scripts\\streamlit.exe run chatbot.py
"""

import hmac
import os

import streamlit as st

import loaders
import rag_engine

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

BOT_NAME = "Ask My Docs"
BOT_TAGLINE = "Ask me anything about our information."

INDEX_NAME = "ask-my-docs"
# Everyone shares one knowledge base. "default" is the same one the full
# version (app.py) uses, so information added there shows up here too.
NAMESPACE = "default"

DEFAULT_ADMIN_ID = "6002604486"
DEFAULT_ADMIN_PASSWORD = "change-me"

st.set_page_config(
    page_title=BOT_NAME,
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Mobile polish: hide Streamlit's own chrome, give the chat room to breathe,
# and keep the input comfortably tappable on a phone.
st.markdown(
    """
    <style>
      #MainMenu, footer, header {visibility: hidden;}

      .block-container {
          padding-top: 2.2rem;
          padding-bottom: 6rem;
          max-width: 46rem;
      }

      /* Readable on a small screen without zooming */
      .stChatMessage p, .stChatMessage li {
          font-size: 1.03rem;
          line-height: 1.6;
      }

      /* Keep the typing box clear of the phone's home bar */
      .stChatInput {
          padding-bottom: env(safe-area-inset-bottom);
      }

      @media (max-width: 640px) {
          .block-container {
              padding-left: 1rem;
              padding-right: 1rem;
              padding-top: 1.4rem;
          }
          h1 {font-size: 1.6rem !important;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def secret(name, fallback=""):
    """Read from Streamlit secrets, then environment. Ignore placeholder text."""
    value = ""
    try:
        if name in st.secrets:
            value = str(st.secrets[name])
    except Exception:
        pass

    value = (value or os.environ.get(name, "")).strip()

    if not value or "PASTE" in value.upper():
        return fallback

    return value


# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------

gemini_key = secret("GOOGLE_API_KEY")
pinecone_key = secret("PINECONE_API_KEY")

# Fallback for local use: ask once if the keys are not in secrets.
if not (gemini_key and pinecone_key):
    st.title(BOT_NAME)
    st.info("Setup needed. Enter the two API keys once to start this bot.")
    gemini_key = gemini_key or st.text_input("Google Gemini API key", type="password")
    pinecone_key = pinecone_key or st.text_input("Pinecone API key", type="password")
    if not (gemini_key and pinecone_key):
        st.stop()


@st.cache_resource(show_spinner=False)
def connect(g_key, p_key):
    return rag_engine.gemini_client(g_key), rag_engine.pinecone_index(p_key, INDEX_NAME)


try:
    with st.spinner("Starting…"):
        client, index = connect(gemini_key, pinecone_key)
except Exception as error:
    st.title(BOT_NAME)
    st.error(f"Could not start: {error}")
    st.stop()


# --------------------------------------------------------------------------
# The chat — this is all a visitor sees
# --------------------------------------------------------------------------

st.title(BOT_NAME)
st.caption(BOT_TAGLINE)

if "history" not in st.session_state:
    st.session_state.history = []

for role, message in st.session_state.history:
    with st.chat_message(role):
        st.markdown(message)

question = st.chat_input("Type your question…")

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                hits = rag_engine.search(client, index, NAMESPACE, question)
                reply, _ = rag_engine.answer(
                    client, question, hits, st.session_state.history[:-1]
                )
            except Exception as error:
                reply = f"Sorry, something went wrong: {error}"

        st.markdown(reply)

    st.session_state.history.append(("assistant", reply))


# --------------------------------------------------------------------------
# Owner area — hidden at the bottom, closed by default
# --------------------------------------------------------------------------

def signed_in():
    return st.session_state.get("is_owner", False)


st.divider()

with st.expander("Owner login"):
    if not signed_in():
        entered_id = st.text_input("ID", key="login_id")
        entered_password = st.text_input("Password", type="password", key="login_pw")

        if st.button("Sign in"):
            real_id = secret("ADMIN_ID", DEFAULT_ADMIN_ID)
            real_password = secret("ADMIN_PASSWORD", DEFAULT_ADMIN_PASSWORD)

            id_ok = hmac.compare_digest(entered_id.strip(), real_id)
            password_ok = hmac.compare_digest(entered_password, real_password)

            if id_ok and password_ok:
                st.session_state.is_owner = True
                st.rerun()
            else:
                st.error("Wrong ID or password.")

    else:
        stored = rag_engine.stats(index, NAMESPACE)
        st.success(f"Signed in. {stored} pieces of knowledge stored.")

        st.markdown("**Paste information**")
        pasted = st.text_area(
            "Text",
            height=200,
            label_visibility="collapsed",
            placeholder="Paste anything here — FAQs, prices, policies, notes…",
        )
        label = st.text_input("Name this information", value="Notes")

        if st.button("Save", type="primary") and pasted.strip():
            with st.spinner("Saving…"):
                chunks = [
                    {"text": piece, "source": label or "Notes"}
                    for piece in loaders.chunk_text(pasted)
                ]
                count = rag_engine.store_chunks(client, index, NAMESPACE, chunks)
            st.success(f"Saved. Added {count} piece{'' if count == 1 else 's'}.")

        st.markdown("**Or upload files**")
        uploads = st.file_uploader(
            "Files",
            label_visibility="collapsed",
            type=["pdf", "docx", "xlsx", "xlsm", "csv", "txt", "md", "json",
                  "png", "jpg", "jpeg", "webp", "gif", "bmp"],
            accept_multiple_files=True,
        )

        if uploads and st.button("Save files"):
            total = 0
            for upload in uploads:
                try:
                    with st.spinner(f"Reading {upload.name}…"):
                        chunks = loaders.load(client, upload.name, upload.getvalue())
                        total += rag_engine.store_chunks(
                            client, index, NAMESPACE, chunks
                        )
                    st.write(f"✅ {upload.name}")
                except Exception as error:
                    st.write(f"❌ {upload.name} — {error}")
            st.success(f"Saved {total} pieces.")

        if st.button("Sign out"):
            st.session_state.is_owner = False
            st.rerun()
