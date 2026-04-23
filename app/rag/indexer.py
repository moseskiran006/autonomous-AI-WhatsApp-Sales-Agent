"""
PHN Technology WhatsApp Agent — Document Indexer

Loads knowledge base JSON files, chunks them, generates embeddings
via Ollama (nomic-embed-text), and stores in ChromaDB.

Runs on application startup if the index doesn't exist.
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from app.config import get_settings

logger = logging.getLogger(__name__)

# Singleton vectorstore instance
_vectorstore: Optional[Chroma] = None


def _get_embeddings() -> OllamaEmbeddings:
    """Create Ollama embeddings instance."""
    settings = get_settings()
    return OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )


def _load_knowledge_base() -> list[Document]:
    """
    Load all JSON files from the knowledge base directory
    and convert them into LangChain Documents.
    """
    settings = get_settings()
    kb_dir = Path(settings.knowledge_base_dir)
    documents = []

    if not kb_dir.exists():
        logger.warning(f"⚠️ Knowledge base directory not found: {kb_dir}")
        return documents

    for json_file in kb_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            file_category = json_file.stem  # e.g., "courses", "bootcamps", "faqs"

            if isinstance(data, list):
                # List of items (courses, FAQs, etc.)
                for item in data:
                    doc_text = _format_item_as_text(item, file_category)
                    metadata = {
                        "source": json_file.name,
                        "category": file_category,
                        "item_name": item.get("name", item.get("question", "unknown")),
                    }
                    documents.append(Document(page_content=doc_text, metadata=metadata))

            elif isinstance(data, dict):
                # Single document (company info, etc.)
                doc_text = _format_item_as_text(data, file_category)
                metadata = {
                    "source": json_file.name,
                    "category": file_category,
                    "item_name": data.get("name", file_category),
                }
                documents.append(Document(page_content=doc_text, metadata=metadata))

            logger.info(f"📄 Loaded {json_file.name} ({len(data) if isinstance(data, list) else 1} items)")

        except Exception as e:
            logger.error(f"❌ Error loading {json_file}: {e}")

    # Load raw text files
    for txt_file in kb_dir.glob("*.txt"):
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                data = f.read()
            metadata = {
                "source": txt_file.name,
                "category": "raw_document",
                "item_name": txt_file.stem,
            }
            documents.append(Document(page_content=data, metadata=metadata))
            logger.info(f"📄 Loaded {txt_file.name} (Raw Text)")
        except Exception as e:
            logger.error(f"❌ Error loading {txt_file}: {e}")

    return documents


def _format_item_as_text(item: dict, category: str) -> str:
    """
    Convert a knowledge base item (dict) into a readable text format
    suitable for embedding and retrieval.
    """
    lines = []

    if category == "faqs":
        # FAQ format
        lines.append(f"Question: {item.get('question', '')}")
        lines.append(f"Answer: {item.get('answer', '')}")
        if item.get("category"):
            lines.append(f"Category: {item['category']}")

    elif category in ["courses", "bootcamps"]:
        # Course/bootcamp format
        lines.append(f"Course Name: {item.get('name', '')}")
        if item.get("description"):
            lines.append(f"Description: {item['description']}")
        if item.get("duration"):
            lines.append(f"Duration: {item['duration']}")
        if item.get("price"):
            lines.append(f"Price: {item['price']}")
        if item.get("original_price"):
            lines.append(f"Original Price: {item['original_price']}")
        if item.get("prerequisites"):
            lines.append(f"Prerequisites: {item['prerequisites']}")
        if item.get("mode"):
            lines.append(f"Mode: {item['mode']}")
        if item.get("schedule"):
            lines.append(f"Schedule: {item['schedule']}")
        if item.get("instructor"):
            lines.append(f"Instructor: {item['instructor']}")
        if item.get("syllabus"):
            if isinstance(item["syllabus"], list):
                lines.append(f"Syllabus Topics: {', '.join(item['syllabus'])}")
            else:
                lines.append(f"Syllabus: {item['syllabus']}")
        if item.get("highlights"):
            if isinstance(item["highlights"], list):
                lines.append(f"Highlights: {', '.join(item['highlights'])}")
            else:
                lines.append(f"Highlights: {item['highlights']}")
        if item.get("placement_support"):
            lines.append(f"Placement Support: {item['placement_support']}")
        if item.get("certificate"):
            lines.append(f"Certificate: {item['certificate']}")
        if item.get("emi_available"):
            lines.append(f"EMI Available: {item['emi_available']}")
        if item.get("next_batch"):
            lines.append(f"Next Batch: {item['next_batch']}")

    elif category == "internships":
        # Summer Internship format
        lines.append(f"Program Name: {item.get('name', '')}")
        if item.get("type"):
            lines.append(f"Type: {item['type'].title()} Internship")
        if item.get("description"):
            lines.append(f"Description: {item['description']}")
        if item.get("duration"):
            lines.append(f"Duration: {item['duration']}")
        if item.get("price"):
            lines.append(f"Price / Seat Reservation Fee: {item['price']}")
        if item.get("package_value"):
            lines.append(f"Package Value: {item['package_value']}")
        if item.get("seats_per_batch"):
            lines.append(f"Seats Per Batch: {item['seats_per_batch']} seats only")
        if item.get("seats_per_campus"):
            lines.append(f"Seats Per Campus: {item['seats_per_campus']} seats only")
        if item.get("mode"):
            lines.append(f"Mode: {item['mode']}")
        if item.get("schedule"):
            lines.append(f"Schedule: {item['schedule']}")
        if item.get("how_it_works"):
            lines.append(f"How It Works: {' | '.join(item['how_it_works'])}")
        # Domains (online internship)
        if item.get("domains"):
            domain_names = []
            for d in item["domains"]:
                if isinstance(d, dict):
                    desc = f"{d.get('name', '')} — {d.get('description', '')}"
                    domain_names.append(desc)
                else:
                    domain_names.append(str(d))
            lines.append(f"Available Domains: {' | '.join(domain_names)}")
        # Campuses (offline internship)
        if item.get("campuses"):
            campus_list = []
            for c in item["campuses"]:
                if isinstance(c, dict):
                    domains_str = ", ".join(c.get("domains", []))
                    campus_list.append(f"{c.get('name', '')} ({c.get('location', '')}) — Domains: {domains_str}")
                else:
                    campus_list.append(str(c))
            lines.append(f"Available Campuses: {' | '.join(campus_list)}")
        if item.get("highlights"):
            if isinstance(item["highlights"], list):
                lines.append(f"Highlights: {', '.join(item['highlights'])}")
            else:
                lines.append(f"Highlights: {item['highlights']}")
        if item.get("certificate"):
            lines.append(f"Certificate: {item['certificate']}")
        if item.get("url"):
            lines.append(f"More Info / Enroll: {item['url']}")

    elif category == "company_info":
        # Company info format — handles nested dicts and lists of dicts
        for key, value in item.items():
            label = key.replace("_", " ").title()
            if isinstance(value, dict):
                lines.append(f"{label}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"  {sub_key.replace('_', ' ').title()}: {sub_value}")
            elif isinstance(value, list):
                # List of plain strings
                if all(isinstance(v, str) for v in value):
                    lines.append(f"{label}: {', '.join(value)}")
                # List of dicts (e.g., services, milestones)
                elif all(isinstance(v, dict) for v in value):
                    lines.append(f"{label}:")
                    for v in value:
                        entry_parts = []
                        for k2, v2 in v.items():
                            entry_parts.append(f"{k2.replace('_',' ').title()}: {v2}")
                        lines.append(f"  - {' | '.join(entry_parts)}")
                else:
                    lines.append(f"{label}: {', '.join(str(v) for v in value)}")
            else:
                lines.append(f"{label}: {value}")

    else:
        # Generic format
        for key, value in item.items():
            lines.append(f"{key.replace('_', ' ').title()}: {value}")

    return "\n".join(lines)


def index_knowledge_base(force_reindex: bool = False) -> Chroma:
    """
    Index the knowledge base into ChromaDB.

    Args:
        force_reindex: If True, delete existing index and rebuild.

    Returns:
        ChromaDB vectorstore instance.
    """
    global _vectorstore
    settings = get_settings()
    persist_dir = settings.chroma_persist_dir

    embeddings = _get_embeddings()

    # Check if index already exists
    if not force_reindex and os.path.exists(persist_dir) and os.listdir(persist_dir):
        logger.info("📦 Loading existing ChromaDB index...")
        _vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="phn_knowledge",
        )
        # Verify it has documents
        collection = _vectorstore._collection
        count = collection.count()
        if count > 0:
            logger.info(f"✅ ChromaDB loaded with {count} documents")
            return _vectorstore
        else:
            logger.info("⚠️ ChromaDB exists but empty, re-indexing...")

    # Load and index documents
    logger.info("🔄 Indexing knowledge base into ChromaDB...")
    documents = _load_knowledge_base()

    if not documents:
        logger.warning("⚠️ No documents found in knowledge base!")
        _vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="phn_knowledge",
        )
        return _vectorstore

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", ", ", " "],
    )
    chunks = text_splitter.split_documents(documents)

    logger.info(f"📋 Split {len(documents)} documents into {len(chunks)} chunks")

    # Create vectorstore with embeddings
    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="phn_knowledge",
    )

    logger.info(f"✅ Indexed {len(chunks)} chunks into ChromaDB")
    return _vectorstore


def get_vectorstore() -> Chroma:
    """Get the singleton vectorstore instance."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = index_knowledge_base()
    return _vectorstore
