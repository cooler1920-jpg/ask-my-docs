"""
Ask My Docs - multi company version.

Three kinds of people use this one app:

  1. A CUSTOMER opens a company's own link, e.g.  ?c=big-bites
     They see one thing: a chat box. Answers come only from that company's
     data. They never see any other company's information.

  2. A COMPANY OWNER signs in with their phone number and password.
     They feed in their own data and copy their own link to share.

  3. THE SUPER ADMIN (you) signs in with the admin number.
     Creates company accounts, switches them off, changes their passwords.

Run:  .venv\\Scripts\\streamlit.exe run saas.py
"""

import os

import streamlit as st

import accounts
import loaders
import rag_engine

PLATFORM_NAME = "Ask My Docs"
INDEX_NAME = "ask-my-docs"
DEFAULT_APP_URL = "https://hh7-ask-my-docs.streamlit.app"

st.set_page_config(
    page_title=PLATFORM_NAME,
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      #MainMenu, footer, header {visibility: hidden;}
      .block-container {padding-top: 2.2rem; padding-bottom: 6rem; max-width: 46rem;}
      .stChatMessage p, .stChatMessage li {font-size: 1.03rem; line-height: 1.6;}
      .stChatInput {padding-bottom: env(safe-area-inset-bottom);}
      @media (max-width: 640px) {
          .block-container {padding-left: 1rem; padding-right: 1rem; padding-top: 1.4rem;}
          h1 {font-size: 1.6rem !important;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Settings and connection
# --------------------------------------------------------------------------

def secret(name, fallback=""):
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


APP_URL = secret("APP_URL", DEFAULT_APP_URL).rstrip("/")
SUPER_ADMIN_ID = accounts.clean_phone(secret("ADMIN_ID", "6002604486"))
SUPER_ADMIN_PASSWORD = secret("ADMIN_PASSWORD", "")

gemini_key = secret("GOOGLE_API_KEY")
pinecone_key = secret("PINECONE_API_KEY")

if not (gemini_key and pinecone_key):
    st.title(PLATFORM_NAME)
    st.error("This app is not set up yet. The administrator needs to add the API keys.")
    st.stop()


@st.cache_resource(show_spinner=False)
def connect(g_key, p_key):
    return rag_engine.gemini_client(g_key), rag_engine.pinecone_index(p_key, INDEX_NAME)


try:
    with st.spinner("Starting…"):
        client, index = connect(gemini_key, pinecone_key)
except Exception as error:
    st.title(PLATFORM_NAME)
    st.error(f"Could not start: {error}")
    st.stop()


def company_link(slug):
    return f"{APP_URL}/?c={slug}"


# --------------------------------------------------------------------------
# 1. Customer view - reached through a company's own link
# --------------------------------------------------------------------------

def customer_view(account):
    st.title(account["company"])
    st.caption("Ask me anything about our information.")

    namespace = accounts.data_namespace(account["slug"])

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
                    hits = rag_engine.search(client, index, namespace, question)
                    reply, _ = rag_engine.answer(
                        client, question, hits, st.session_state.history[:-1]
                    )
                except Exception as error:
                    reply = f"Sorry, something went wrong: {error}"

            st.markdown(reply)

        st.session_state.history.append(("assistant", reply))


# --------------------------------------------------------------------------
# 2. Feeding data - used by a company owner and by the super admin
# --------------------------------------------------------------------------

def knowledge_panel(namespace, heading="Your information"):
    stored = rag_engine.stats(index, namespace)
    st.subheader(heading)
    st.caption(f"{stored} pieces of knowledge stored.")

    st.markdown("**Paste information**")
    pasted = st.text_area(
        "Text",
        height=180,
        label_visibility="collapsed",
        placeholder="Prices, opening hours, policies, product details, FAQs…",
    )
    label = st.text_input("Name this information", value="Notes")

    if st.button("Save text", type="primary") and pasted.strip():
        with st.spinner("Saving…"):
            chunks = [
                {"text": piece, "source": label or "Notes"}
                for piece in loaders.chunk_text(pasted)
            ]
            count = rag_engine.store_chunks(client, index, namespace, chunks)
        st.success(f"Saved {count} piece{'' if count == 1 else 's'}.")

    st.markdown("**Or upload files**")
    st.caption("PDF · Word · Excel · CSV · TXT · photos")

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
                    total += rag_engine.store_chunks(client, index, namespace, chunks)
                st.write(f"✅ {upload.name}")
            except Exception as error:
                st.write(f"❌ {upload.name} — {error}")
        st.success(f"Saved {total} pieces.")


# --------------------------------------------------------------------------
# 3. Company owner view
# --------------------------------------------------------------------------

def owner_view(account):
    st.title(account["company"])
    st.caption(f"Signed in as {account['phone']}")

    link = company_link(account["slug"])
    st.markdown("**Your link — share this with your customers**")
    st.code(link, language=None)
    st.caption(
        "Put it on your website, in an advertisement, or send it on WhatsApp. "
        "Anyone who opens it can ask questions about your information only."
    )

    st.divider()
    knowledge_panel(accounts.data_namespace(account["slug"]))

    st.divider()
    with st.expander("Try it yourself"):
        st.caption("Ask a question the way your customer would.")
        trial = st.text_input("Question", key="owner_trial")
        if st.button("Ask") and trial.strip():
            namespace = accounts.data_namespace(account["slug"])
            with st.spinner("Thinking…"):
                hits = rag_engine.search(client, index, namespace, trial)
                reply, _ = rag_engine.answer(client, trial, hits)
            st.info(reply)

    if st.button("Sign out"):
        st.session_state.clear()
        st.rerun()


# --------------------------------------------------------------------------
# 4. Super admin view
# --------------------------------------------------------------------------

def admin_view():
    st.title("Administrator")
    st.caption("You control every company account on this platform.")

    every_account = accounts.list_accounts(index)

    st.subheader("Add a company")
    with st.form("new_company", clear_on_submit=True):
        company = st.text_input("Company name", placeholder="Big Bites Restaurant")
        phone = st.text_input("Their phone number", placeholder="9876543210")
        password = st.text_input("Password to give them", type="password")
        submitted = st.form_submit_button("Create account", type="primary")

    if submitted:
        try:
            created = accounts.create_account(index, phone, company, password)
            # Remember what to show, then rerun so the list below includes the
            # new company instead of the version read before it was saved.
            st.session_state.new_account = {
                "company": created["company"],
                "phone": created["phone"],
                "password": password,
                "slug": created["slug"],
            }
            st.rerun()
        except ValueError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Could not create the account: {error}")

    # Shown once, right after creation - this is the only time the password
    # is visible, because only its hash is stored.
    if "new_account" in st.session_state:
        fresh = st.session_state.pop("new_account")
        st.success(f"Created {fresh['company']}.")
        st.markdown("**Give them these three things:**")
        st.code(
            f"Website:  {APP_URL}\n"
            f"Login:    {fresh['phone']}\n"
            f"Password: {fresh['password']}",
            language=None,
        )
        st.caption(f"Their customer link: {company_link(fresh['slug'])}")
        st.warning("Copy the password now — it is not shown again.")

    st.divider()
    st.subheader(f"Companies ({len(every_account)})")

    if not every_account:
        st.caption("No companies yet. Add one above.")

    for account in every_account:
        namespace = accounts.data_namespace(account["slug"])
        state = "🟢 Active" if account["active"] else "🔴 Switched off"

        with st.expander(f"{account['company']} — {account['phone']} — {state}"):
            st.caption(f"Joined {account['created']}")
            st.code(company_link(account["slug"]), language=None)
            st.caption(f"{rag_engine.stats(index, namespace)} pieces of knowledge stored")

            columns = st.columns(2)

            with columns[0]:
                if account["active"]:
                    if st.button("Switch off", key=f"off_{account['phone']}"):
                        accounts.set_active(index, account["phone"], False)
                        st.rerun()
                else:
                    if st.button("Switch back on", key=f"on_{account['phone']}"):
                        accounts.set_active(index, account["phone"], True)
                        st.rerun()

            with columns[1]:
                new_password = st.text_input(
                    "New password",
                    type="password",
                    key=f"pw_{account['phone']}",
                )
                if st.button("Change password", key=f"chpw_{account['phone']}"):
                    try:
                        accounts.change_password(index, account["phone"], new_password)
                        st.success("Password changed.")
                    except ValueError as error:
                        st.error(str(error))

            st.divider()
            st.caption(
                "Deleting removes their login **and** everything they uploaded. "
                "This cannot be undone."
            )
            confirm = st.text_input(
                f"Type DELETE to remove {account['company']}",
                key=f"del_{account['phone']}",
            )
            if st.button("Delete this company", key=f"delbtn_{account['phone']}"):
                if confirm.strip().upper() == "DELETE":
                    rag_engine.wipe(index, namespace)
                    accounts.delete_account(index, account["phone"])
                    st.success(f"{account['company']} removed.")
                    st.rerun()
                else:
                    st.error("Type DELETE in the box to confirm.")

    st.divider()
    if st.button("Sign out"):
        st.session_state.clear()
        st.rerun()


# --------------------------------------------------------------------------
# 5. Sign in
# --------------------------------------------------------------------------

def sign_in_view():
    st.title(PLATFORM_NAME)
    st.caption("Sign in to manage your company's chatbot.")

    phone = st.text_input("Phone number")
    password = st.text_input("Password", type="password")

    if st.button("Sign in", type="primary"):
        clean = accounts.clean_phone(phone)

        # The super admin is configured in secrets, not stored as a company.
        if clean and clean == SUPER_ADMIN_ID:
            if SUPER_ADMIN_PASSWORD and password == SUPER_ADMIN_PASSWORD:
                st.session_state.role = "admin"
                st.rerun()
            else:
                st.error("Wrong phone number or password.")
            return

        try:
            account = accounts.sign_in(index, clean, password)
            st.session_state.role = "owner"
            st.session_state.phone = account["phone"]
            st.rerun()
        except ValueError as error:
            st.error(str(error))

    st.divider()
    st.caption(
        "Are you a customer? Use the link your company gave you — "
        "you do not need to sign in."
    )


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

requested_company = st.query_params.get("c", "")

if requested_company:
    account = accounts.find_by_slug(index, requested_company.strip().lower())

    if not account:
        st.title(PLATFORM_NAME)
        st.error("This link is not valid. Please check it with the company that gave it to you.")
    elif not account["active"]:
        st.title(account["company"])
        st.warning("This chatbot is currently unavailable. Please try again later.")
    else:
        customer_view(account)

elif st.session_state.get("role") == "admin":
    admin_view()

elif st.session_state.get("role") == "owner":
    account = accounts.get_account(index, st.session_state.get("phone", ""))
    if account and account["active"]:
        owner_view(account)
    else:
        st.session_state.clear()
        st.warning("Your session has ended. Please sign in again.")
        sign_in_view()

else:
    sign_in_view()
