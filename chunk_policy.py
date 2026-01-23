import os
import re
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader

import pinecone
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

# =========================
# Load policy text
# =========================

def load_policy_text() -> str:
    reader = PdfReader(POLICY_FILE)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# =========================
# Section-aware splitter
# =========================

def split_policy(text: str):
    section_pattern = r"\n(?=\d+\.\s)"
    sections = re.split(section_pattern, text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks = []

    for sec in sections:
        header_match = re.match(r"(\d+\.\s[^\n]+)", sec)
        header = header_match.group(1) if header_match else "General Policy"

        for i, chunk in enumerate(splitter.split_text(sec)):
            chunks.append({
                "id": f"{header}-{i}",
                "text": chunk.strip(),
                "metadata": {
                    "section": header,
                    "chunk_id": i
                }
            })

    return chunks

# =========================
# Pinecone upload (DIRECT SDK)
# =========================

def upload_chunks(chunks):
    pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=pinecone.ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    index = pc.Index(INDEX_NAME)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBED_DIM
    )

    vectors = []

    for chunk in chunks:
        vector = embeddings.embed_query(chunk["text"])
        vectors.append((
            chunk["id"],
            vector,
            chunk["metadata"]
        ))

    index.upsert(vectors=vectors)

    print(f"Uploaded {len(vectors)} policy chunks to Pinecone")

# =========================
# Main
# =========================

def main():
    print("Loading policy document...")
    text = load_policy_text()

    print("Chunking policy...")
    chunks = split_policy(text)

    print("Uploading to Pinecone...")
    upload_chunks(chunks)

    print("Done.")

if __name__ == "__main__":
    main()
