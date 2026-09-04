# Ask My Docs — a private AI chatbot for your own information

Upload your documents, spreadsheets and photos. Ask questions in plain language.
The bot answers **only** from what you gave it — if the answer is not in your
files, it says so instead of making something up.

Built by **HH7 Softech Solution (OPC) Pvt Ltd**.

---

## What it can read

| Type | Files | How it works |
|---|---|---|
| Documents | `.pdf` `.docx` `.txt` `.md` | Text is pulled straight out of the file |
| **Scanned PDFs** | `.pdf` with no text layer | Automatically read by Gemini, like a person reading a photocopy |
| Spreadsheets | `.xlsx` `.csv` | Every sheet, row and column becomes searchable text |
| **Photos** | `.jpg` `.png` `.webp` | Gemini describes the image **and reads any text in it**. You can add your own explanation, which is stored alongside |
| Typed text | paste anything | FAQs, policies, price lists, notes |

---

## How it works (the short version)

1. **Chop** — your file is split into small overlapping pieces.
2. **Embed** — each piece is turned into a list of numbers (a *vector*) by
   Google's `gemini-embedding-001`. Pieces about similar topics get similar numbers.
3. **Store** — the vectors go into **Pinecone**, a vector database.
4. **Ask** — your question is turned into a vector too, and Pinecone finds the
   handful of pieces that are closest to it.
5. **Answer** — only those pieces are handed to `gemini-2.5-flash`, with a strict
   instruction: *answer from this material or admit you don't know.*

That is **RAG** — Retrieval Augmented Generation. The AI is not "trained" on your
data; it is *shown* the right page at the right moment.

---

## Setup (one time, about 5 minutes)

### 1. Get two free API keys

| Key | Where | Cost |
|---|---|---|
| Google Gemini | https://aistudio.google.com/apikey | Free tier |
| Pinecone | https://app.pinecone.io | Free tier |

### 2. Save them

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste
your keys in. That file is already blocked by `.gitignore`, so it can never be
pushed to GitHub by accident.

### 3. Install and check

```bash
.venv\Scripts\python.exe selftest.py
```

This stores three test facts, asks three questions, and confirms the bot refuses
to answer things it was not told. It cleans up after itself.

### 4. Run it

```bash
.venv\Scripts\streamlit.exe run app.py
```

Opens at http://localhost:8501

---

## Putting it online (free public link)

1. Push this folder to a **GitHub** repo.
2. Go to https://share.streamlit.io, click **New app**, choose the repo,
   set the main file to `app.py`.
3. Open **Advanced settings → Secrets** and paste:

   ```toml
   GOOGLE_API_KEY = "your-key"
   PINECONE_API_KEY = "your-key"
   ```

4. Deploy. You get a public link like `https://your-app.streamlit.app`.

> **Never** commit `secrets.toml`. Always paste keys into Streamlit's Secrets box.

---

## Knowledge bases

The **Knowledge base** box in the sidebar keeps different sets of documents apart.
Type `zafari` and upload the Zafari manual; type `hr-policy` and upload HR
documents. The bot only ever searches the one currently selected. One deployment,
many separate bots.

---

## Files

| File | What it does |
|---|---|
| `app.py` | The screen you see — upload area and chat |
| `rag_engine.py` | Embeddings, Pinecone storage and search, answering |
| `loaders.py` | Turns each file type into plain text |
| `selftest.py` | Proves the whole pipeline works |
| `requirements.txt` | The libraries needed |

---

## If something goes wrong

| Message | Fix |
|---|---|
| `Could not connect` | A key is wrong or has a stray space. Re-copy it. |
| `No readable text found` | The file is empty or password protected. |
| Bot says it cannot find the answer | The information was never uploaded, or you are in a different knowledge base. Check the sidebar counter. |
| Index takes a minute on first run | Normal — Pinecone is creating it. Only happens once. |
