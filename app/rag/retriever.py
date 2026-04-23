"""
PHN Technology WhatsApp Agent — RAG Retriever

Performs semantic search over the ChromaDB vectorstore
to find relevant course/workshop/FAQ information.
"""

import logging
from typing import Optional
from langchain_core.documents import Document

from app.rag.indexer import get_vectorstore

logger = logging.getLogger(__name__)


def retrieve_documents(
    query: str,
    k: int = 4,
    category: Optional[str] = None,
) -> list[Document]:
    """
    Retrieve relevant documents from ChromaDB.

    Args:
        query: Search query (user's message)
        k: Number of documents to retrieve
        category: Optional category filter (e.g., "courses", "bootcamps", "faqs")

    Returns:
        List of relevant Document objects with content and metadata.
    """
    try:
        vectorstore = get_vectorstore()

        # Build search kwargs
        search_kwargs = {"k": k}

        if category:
            search_kwargs["filter"] = {"category": category}

        # Perform similarity search
        docs = vectorstore.similarity_search(
            query=query,
            **search_kwargs,
        )

        logger.info(
            f"🔍 Retrieved {len(docs)} docs for query: '{query[:50]}...' "
            f"(category: {category or 'all'})"
        )

        return docs

    except Exception as e:
        logger.error(f"❌ Retrieval failed: {e}")
        return []


def retrieve_with_scores(
    query: str,
    k: int = 4,
    score_threshold: float = 0.5,
) -> list[tuple[Document, float]]:
    """
    Retrieve documents with relevance scores.
    Useful for debugging and quality monitoring.

    Args:
        query: Search query
        k: Number of documents to retrieve
        score_threshold: Minimum similarity score (0-1, lower is more similar in ChromaDB)

    Returns:
        List of (Document, score) tuples sorted by relevance.
    """
    try:
        vectorstore = get_vectorstore()

        results = vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
        )

        # Filter by threshold
        filtered = [(doc, score) for doc, score in results if score >= score_threshold]

        logger.info(
            f"🔍 Retrieved {len(filtered)}/{len(results)} docs above threshold "
            f"(threshold: {score_threshold})"
        )

        return filtered

    except Exception as e:
        logger.error(f"❌ Scored retrieval failed: {e}")
        return []
