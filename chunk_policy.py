import os
import re
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# Environment
# =========================

load_dotenv(override=True)

# =========================
# Config
# =========================

BASE_DIR = Path(__file__).parent
POLICY_FILE = BASE_DIR / "docs" / "Generic E-Commerce Company Master Policy Compendium.pdf"

INDEX_NAME = "project-6-ecommerce-agent"

EMBEDDING_MODEL = "text-embedding-3-large"
EMBED_DIM = 1024

# Similarity floor for weak-policy documents
MIN_SCORE = 0.045

REFUND_KEYWORDS = [
    "refund",
    "refunds",
    "return",
    "returns",
    "exchange",
    "chargeback",
    "reimbursement",
    "money back",
    "cancel",
    "cancellation"
]

# =========================
# PDF Loading
# =========================

def load_policy_text() -> str:
    reader = PdfReader(POLICY_FILE)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)

# =========================
# Chunking Helpers
# =========================

def has_refund_signal(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in REFUND_KEYWORDS)

def is_useful_length(text: str) -> bool:
    return len(text.split()) >= 40

# =========================
# Chunk + Upload (INGESTION)
# =========================

def chunk_and_upload_policy():
    print("Loading policy document...")
    text = load_policy_text()

    print("Chunking policy (weak refund signal)...")
    sections = re.split(r"\n(?=\d+\.\s)", text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", "; ", " "]
    )

    chunks = []

    for sec in sections:
        header_match = re.match(r"(\d+\.\s[^\n]+)", sec)
        section_header = header_match.group(1).strip() if header_match else "General Policy"

        body = sec
        body = re.sub(r"\d+\.\s[^\n]+", "", body, count=1)
        body = re.sub(r"Page\s+\d+\s+of\s+\d+", "", body, flags=re.IGNORECASE)
        body = body.strip()

        for i, chunk in enumerate(splitter.split_text(body)):
            if not has_refund_signal(chunk):
                continue
            if not is_useful_length(chunk):
                continue

            chunks.append({
                "id": f"{section_header}-{i}",
                "text": chunk.strip(),
                "metadata": {
                    "section": section_header,
                    "chunk_id": i,
                    "policy_type": "refund",
                    "signal_strength": "weak"
                }
            })

    if not chunks:
        print("⚠️ No refund-related chunks found.")
        return

    print(f"Prepared {len(chunks)} chunks")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec={
                "serverless": {
                    "cloud": "aws",
                    "region": "us-east-1"
                }
            }
        )

    index = pc.Index(INDEX_NAME)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBED_DIM
    )

    vectors = []
    for chunk in chunks:
        vector = embeddings.embed_query(chunk["text"])
        vectors.append((chunk["id"], vector, chunk["metadata"]))

    index.upsert(vectors=vectors)

    print(f"Uploaded {len(vectors)} refund-related policy chunks")

# =========================
# Retriever (RUNTIME)
# =========================

def retrieve_refund_policy(query: str, top_k: int = 10) -> dict:
    """
    Refund-aware retriever for weak-signal policy documents.
    """

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBED_DIM
    )

    query_vector = embeddings.embed_query(query)

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter={"policy_type": "refund"}
    )

    matches = [
        m for m in results.get("matches", [])
        if m.get("score", 0) >= MIN_SCORE
    ]

    return {
        "query": query,
        "policy_signal": "weak",
        "matches": matches,
        "explanation": (
            "Refunds are referenced only in limited contexts such as "
            "customer service recovery and fraud mitigation. "
            "The policy does not define explicit refund timelines or guarantees."
        )
    }

# =========================
# CLI Usage
# =========================

if __name__ == "__main__":
    import sys

    if "--ingest" in sys.argv:
        chunk_and_upload_policy()
    else:
        test_query = "Can I get a refund after 30 days?"
        result = retrieve_refund_policy(test_query)

        print("\nQuery:", result["query"])
        print("Policy Signal:", result["policy_signal"])
        print("Explanation:", result["explanation"])
        print("\nMatches:\n")

        for i, match in enumerate(result["matches"], start=1):
            print(f"{i}. Score: {match['score']:.4f}")
            print(f"   Section: {match['metadata'].get('section')}")
            print(f"   Chunk ID: {match['metadata'].get('chunk_id')}")
            print()
