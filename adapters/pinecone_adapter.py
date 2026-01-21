import os
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

def fetch_from_pinecone(item_category: str) -> dict:
    """
    Lazy-loaded Pinecone policy retrieval.
    Pinecone is only touched when this function is called.
    """

    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get(
        "PINECONE_INDEX_NAME",
        "project-6-ecommerce-agent"
    )

    pc = Pinecone(api_key=api_key)

    # THIS is the critical part: inside the function
    index = pc.Index(index_name)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=1024
    )

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

    results = vectorstore.similarity_search(
        query=item_category,
        k=1
    )

    if not results:
        return {"policy_clause": "No policy found for this category."}

    return {"policy_clause": results[0].page_content}
