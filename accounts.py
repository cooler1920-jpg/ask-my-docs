"""
Account storage for the multi-company version.

Every company gets:
  - a phone number, which is their login
  - a password (stored hashed, never in plain text)
  - a short name for their link, e.g. bigbites -> ?c=bigbites
  - their own separate data area, so one company can never see another's

Accounts are kept in the Pinecone index in a reserved area called
"__accounts__". Storing them there means no second service and no extra
password to manage.
"""

import hashlib
import secrets
import time

import rag_engine

ACCOUNTS_NS = "__accounts__"

# Pinecone will not accept an all-zero vector, and we never search these by
# meaning anyway - they are looked up by id. So every account row uses the
# same placeholder vector.
PLACEHOLDER = [1.0] + [0.0] * (rag_engine.EMBED_DIM - 1)

PBKDF2_ROUNDS = 200_000


# --------------------------------------------------------------------------
# Passwords
# --------------------------------------------------------------------------

def hash_password(password, salt=None):
    """Return (salt, hash). The plain password is never stored."""
    if salt is None:
        salt = secrets.token_hex(16)

    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ROUNDS
    )
    return salt, digest.hex()


def password_matches(password, salt, expected_hash):
    _, actual = hash_password(password, salt)
    return secrets.compare_digest(actual, expected_hash)


# --------------------------------------------------------------------------
# Names and phone numbers
# --------------------------------------------------------------------------

def clean_phone(phone):
    return "".join(ch for ch in str(phone) if ch.isdigit())


def make_slug(name):
    """Turn 'Big Bites Restaurant' into 'big-bites-restaurant'."""
    kept = [ch.lower() if ch.isalnum() else "-" for ch in str(name)]
    slug = "".join(kept)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:40] or "company"


def data_namespace(slug):
    """Where this company's documents live. Kept apart from every other."""
    return f"co-{slug}"


# --------------------------------------------------------------------------
# Reading and writing accounts
# --------------------------------------------------------------------------

def _row_to_account(row):
    meta = row.get("metadata") or {}
    return {
        "phone": meta.get("phone", ""),
        "company": meta.get("company", ""),
        "slug": meta.get("slug", ""),
        "salt": meta.get("salt", ""),
        "hash": meta.get("hash", ""),
        "active": bool(meta.get("active", True)),
        "created": meta.get("created", ""),
    }


def get_account(index, phone):
    phone = clean_phone(phone)
    if not phone:
        return None

    try:
        result = index.fetch(ids=[phone], namespace=ACCOUNTS_NS)
    except Exception:
        return None

    vectors = result.get("vectors") or {}
    row = vectors.get(phone)
    return _row_to_account(row) if row else None


def find_by_slug(index, slug):
    for account in list_accounts(index):
        if account["slug"] == slug:
            return account
    return None


def list_accounts(index):
    """Every company account, newest first."""
    try:
        result = index.query(
            vector=PLACEHOLDER,
            top_k=500,
            include_metadata=True,
            namespace=ACCOUNTS_NS,
        )
    except Exception:
        return []

    accounts = [_row_to_account(m) for m in result.get("matches", [])]
    return sorted(accounts, key=lambda a: a["created"], reverse=True)


def save_account(index, account):
    index.upsert(
        vectors=[{
            "id": account["phone"],
            "values": PLACEHOLDER,
            "metadata": {
                "phone": account["phone"],
                "company": account["company"],
                "slug": account["slug"],
                "salt": account["salt"],
                "hash": account["hash"],
                "active": account["active"],
                "created": account["created"],
            },
        }],
        namespace=ACCOUNTS_NS,
    )


def create_account(index, phone, company, password):
    """
    Add a new company. Raises ValueError with a plain-English reason if the
    details are not usable.
    """
    phone = clean_phone(phone)
    company = company.strip()

    if len(phone) < 10:
        raise ValueError("Phone number must have at least 10 digits.")
    if not company:
        raise ValueError("Company name cannot be empty.")
    if len(password) < 4:
        raise ValueError("Password must be at least 4 characters.")
    if get_account(index, phone):
        raise ValueError(f"{phone} already has an account.")

    slug = make_slug(company)
    if find_by_slug(index, slug):
        slug = f"{slug}-{phone[-4:]}"

    salt, digest = hash_password(password)

    account = {
        "phone": phone,
        "company": company,
        "slug": slug,
        "salt": salt,
        "hash": digest,
        "active": True,
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }
    save_account(index, account)
    return account


def set_active(index, phone, active):
    account = get_account(index, phone)
    if not account:
        raise ValueError("No such account.")
    account["active"] = bool(active)
    save_account(index, account)
    return account


def change_password(index, phone, new_password):
    account = get_account(index, phone)
    if not account:
        raise ValueError("No such account.")
    if len(new_password) < 4:
        raise ValueError("Password must be at least 4 characters.")

    account["salt"], account["hash"] = hash_password(new_password)
    save_account(index, account)
    return account


def delete_account(index, phone):
    """Remove the login. The company's documents are deleted separately."""
    index.delete(ids=[clean_phone(phone)], namespace=ACCOUNTS_NS)


def sign_in(index, phone, password):
    """
    Return the account on success. Raises ValueError with the reason otherwise.
    """
    account = get_account(index, phone)
    if not account or not password_matches(password, account["salt"], account["hash"]):
        raise ValueError("Wrong phone number or password.")
    if not account["active"]:
        raise ValueError("This account has been switched off. Contact the administrator.")
    return account
