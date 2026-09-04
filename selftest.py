"""
Self test — proves your API keys work and the whole pipeline runs.

Run it AFTER you put your keys in .streamlit/secrets.toml:

    .venv\\Scripts\\python.exe selftest.py

It stores three facts, asks two questions, then deletes the test data.
Nothing you own is touched — it uses a throwaway knowledge base called "selftest".
"""

import os
import sys
import tomllib

import loaders
import rag_engine

NAMESPACE = "selftest"
INDEX = "ask-my-docs"

FACTS = [
    "HH7 Softech Solution (OPC) Private Limited was incorporated on 13 July 2026. "
    "Its CIN is U62013AS2026OPC030826 and its founder is Bishal Malakar.",
    "Zafari is a travel app for India. It is live on the Google Play Store under "
    "the package name com.bishal.zafari. Premium plans cost 39 rupees monthly, "
    "99 rupees quarterly and 299 rupees yearly.",
    "Zayka is a QR based restaurant ordering system. Six restaurants use it, "
    "including Big Bites, Lokenath, MAA ANNAPURNA and Kutum.",
]

QUESTIONS = [
    ("What is the CIN of HH7 Softech Solution?", "U62013AS2026OPC030826"),
    ("How much does the Zafari yearly plan cost?", "299"),
    ("Who is the Prime Minister of Japan?", None),  # must refuse — not in the data
]


def load_keys():
    google = os.environ.get("GOOGLE_API_KEY", "")
    pine = os.environ.get("PINECONE_API_KEY", "")

    path = os.path.join(".streamlit", "secrets.toml")
    if os.path.exists(path):
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
        google = google or data.get("GOOGLE_API_KEY", "")
        pine = pine or data.get("PINECONE_API_KEY", "")

    return google, pine


def main():
    google_key, pinecone_key = load_keys()

    if not google_key or not pinecone_key:
        print("MISSING KEYS.")
        print("Copy .streamlit/secrets.toml.example to .streamlit/secrets.toml")
        print("and paste your two keys into it, then run this again.")
        return 1

    print("1/5  Connecting to Gemini...")
    client = rag_engine.gemini_client(google_key)

    print("2/5  Connecting to Pinecone (creates the index on first run, ~1 min)...")
    index = rag_engine.pinecone_index(pinecone_key, INDEX)

    print("3/5  Storing 3 test facts...")
    rag_engine.wipe(index, NAMESPACE)
    chunks = [{"text": fact, "source": "selftest"} for fact in FACTS]
    stored = rag_engine.store_chunks(client, index, NAMESPACE, chunks)
    print(f"     stored {stored} pieces")

    print("4/5  Asking questions...")
    failures = 0
    for question, expected in QUESTIONS:
        hits = rag_engine.search(client, index, NAMESPACE, question)
        reply, sources = rag_engine.answer(client, question, hits)
        print(f"\n     Q: {question}")
        print(f"     A: {reply}")

        if expected is None:
            if "could not find" in reply.lower():
                print("     PASS - correctly refused to invent an answer")
            else:
                print("     FAIL - it answered something it should not know")
                failures += 1
        elif expected.lower() in reply.lower():
            print("     PASS")
        else:
            print(f"     FAIL - expected to see '{expected}'")
            failures += 1

    print("\n5/5  Cleaning up test data...")
    rag_engine.wipe(index, NAMESPACE)

    if failures:
        print(f"\nDONE with {failures} failure(s).")
        return 1

    print("\nALL TESTS PASSED. Your chatbot is working. Run:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
