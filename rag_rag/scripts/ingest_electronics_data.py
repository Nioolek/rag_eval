#!/usr/bin/env python
"""
Electronics Product Data Ingestion Script.

Ingests generated FAQ and document data into storage backends.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / "src"))

from rag_rag.storage.faq_store import FAQStore
from rag_rag.storage.vector_store import VectorStore
from rag_rag.storage.fulltext_store import FulltextStore
from rag_rag.services.embedding_service import EmbeddingService, EmbeddingConfig
from rag_rag.core.logging import get_logger
import os

logger = get_logger("rag_rag.ingestion")


class DataIngestionPipeline:
    """Pipeline for ingesting data into all storage backends."""

    def __init__(
        self,
        data_dir: str = "data",
        faq_count: int = 500,
        doc_count: int = 200,
        use_real_embeddings: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.generated_dir = script_dir / "data/generated"
        self.faq_count = faq_count
        self.doc_count = doc_count
        self.use_real_embeddings = use_real_embeddings

        # Storage instances
        self.faq_store: Optional[FAQStore] = None
        self.vector_store: Optional[VectorStore] = None
        self.fulltext_store: Optional[FulltextStore] = None
        self.embedding_service: Optional[EmbeddingService] = None

    async def initialize(self) -> None:
        """Initialize all storage backends."""
        print("Initializing storage backends...")

        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize FAQ Store
        print("  - FAQ Store (SQLite)...")
        self.faq_store = FAQStore(
            db_path=self.data_dir / "faq.db",
            match_threshold=0.85,
        )
        await self.faq_store.initialize()

        # Initialize Vector Store
        print("  - Vector Store (Chroma)...")
        self.vector_store = VectorStore(
            persist_dir=self.data_dir / "chroma",
            collection_name="electronics_docs",
            embedding_dimension=1024,
        )
        await self.vector_store.initialize()

        # Initialize Fulltext Store
        print("  - Fulltext Store (Whoosh)...")
        self.fulltext_store = FulltextStore(
            index_dir=self.data_dir / "whoosh",
            index_name="electronics_docs",
        )
        await self.fulltext_store.initialize()

        # Initialize Embedding Service (if API key available)
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if self.use_real_embeddings and api_key and api_key != "sk-placeholder-key":
            print("  - Embedding Service (DashScope)...")
            self.embedding_service = EmbeddingService(EmbeddingConfig(
                api_key=api_key,
                model="text-embedding-v3",
                dimension=1024,
                batch_size=10,  # DashScope max batch size is 10
            ))
            await self.embedding_service.initialize()
            print("    Using real embeddings from DashScope API")
        else:
            print("  - Embedding Service: SKIPPED (no valid API key)")
            print("    Vector store will use random embeddings (not suitable for production)")

        print("Storage backends initialized!\n")

    async def load_generated_data(self) -> tuple[list[dict], list[dict]]:
        """Load generated FAQ and document data."""
        faqs_file = self.generated_dir / "faqs.json"
        docs_file = self.generated_dir / "documents.json"

        faqs = []
        documents = []

        if faqs_file.exists():
            with open(faqs_file, encoding="utf-8") as f:
                faqs = json.load(f)
            print(f"Loaded {len(faqs)} FAQs from {faqs_file}")
        else:
            print(f"FAQ file not found: {faqs_file}")
            print("Run generate_electronics_data.py first!")

        if docs_file.exists():
            with open(docs_file, encoding="utf-8") as f:
                documents = json.load(f)
            print(f"Loaded {len(documents)} documents from {docs_file}")
        else:
            print(f"Document file not found: {docs_file}")
            print("Run generate_electronics_data.py first!")

        return faqs, documents

    async def ingest_faqs(self, faqs: list[dict]) -> int:
        """Ingest FAQs into FAQ store."""
        if not self.faq_store or not faqs:
            return 0

        print(f"\nIngesting {len(faqs)} FAQs...")

        count = 0
        batch_size = 50

        for i in range(0, len(faqs), batch_size):
            batch = faqs[i : i + batch_size]
            for faq in batch:
                await self.faq_store.set(
                    faq["id"],
                    {
                        "question": faq["question"],
                        "answer": faq["answer"],
                        "category": faq.get("category", ""),
                        "keywords": faq.get("keywords", []),
                        "metadata": {
                            "product": faq.get("product"),
                            "brand": faq.get("brand"),
                        },
                    },
                )
                count += 1

            print(f"  Ingested {count}/{len(faqs)} FAQs...")

        print(f"FAQ ingestion complete: {count} items")
        return count

    async def ingest_documents_vector(self, documents: list[dict]) -> int:
        """Ingest documents into vector store with real embeddings."""
        if not self.vector_store or not documents:
            return 0

        print(f"\nIngesting {len(documents)} documents into Vector Store...")

        import random

        count = 0
        batch_size = 20 if self.embedding_service else 50

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            docs_to_add = []
            for doc in batch:
                docs_to_add.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": {
                        "title": doc.get("title", ""),
                        "category": doc.get("category", ""),
                        "product": doc.get("product"),
                        "brand": doc.get("brand"),
                        "keywords": doc.get("keywords", []),
                    },
                })

            # Generate embeddings
            if self.embedding_service:
                # Use real embeddings from DashScope
                contents = [doc["content"] for doc in batch]
                try:
                    embeddings = await self.embedding_service.embed(contents)
                    print(f"  Generated real embeddings for batch {i//batch_size + 1}")
                except Exception as e:
                    print(f"  Warning: Embedding failed ({e}), using random vectors")
                    embeddings = [[random.uniform(-1, 1) for _ in range(1024)] for _ in batch]
            else:
                # Fallback to random embeddings (not suitable for production)
                embeddings = [[random.uniform(-1, 1) for _ in range(1024)] for _ in batch]

            await self.vector_store.add(docs_to_add, embeddings)
            count += len(docs_to_add)

            print(f"  Ingested {count}/{len(documents)} documents...")

        embedding_type = "real" if self.embedding_service else "random (WARNING: not suitable for production)"
        print(f"Vector Store ingestion complete: {count} documents (embeddings: {embedding_type})")
        return count

    async def ingest_documents_fulltext(self, documents: list[dict]) -> int:
        """Ingest documents into fulltext store."""
        if not self.fulltext_store or not documents:
            return 0

        print(f"\nIngesting {len(documents)} documents into Fulltext Store...")

        count = 0
        batch_size = 50

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            docs_to_add = []
            for doc in batch:
                docs_to_add.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": {
                        "title": doc.get("title", ""),
                        "category": doc.get("category", ""),
                        "product": doc.get("product"),
                        "brand": doc.get("brand"),
                    },
                })

            await self.fulltext_store.add(docs_to_add)
            count += len(docs_to_add)

            print(f"  Ingested {count}/{len(documents)} documents...")

        print(f"Fulltext Store ingestion complete: {count} documents")
        return count

    async def verify_ingestion(self) -> dict[str, Any]:
        """Verify that data was ingested correctly."""
        print("\nVerifying ingestion...")

        stats = {}

        # Check FAQ store
        if self.faq_store:
            faq_keys = await self.faq_store.keys()
            stats["faq_count"] = len(faq_keys)
            print(f"  FAQ Store: {stats['faq_count']} items")

            # Test FAQ search
            if faq_keys:
                first_key = faq_keys[0]
                faq = await self.faq_store.get(first_key)
                if faq:
                    print(f"    Sample FAQ: {faq.get('question', 'N/A')[:50]}...")

        # Check Vector store
        if self.vector_store:
            vector_count = await self.vector_store.count()
            stats["vector_count"] = vector_count
            print(f"  Vector Store: {stats['vector_count']} documents")

        # Check Fulltext store
        if self.fulltext_store:
            fulltext_count = await self.fulltext_store.count()
            stats["fulltext_count"] = fulltext_count
            print(f"  Fulltext Store: {stats['fulltext_count']} documents")

            # Test fulltext search
            if fulltext_count > 0:
                results = await self.fulltext_store.search("iPhone", top_k=3)
                print(f"    Search 'iPhone': {len(results)} results")

        return stats

    async def close(self) -> None:
        """Close all storage backends."""
        print("\nClosing storage backends...")

        if self.faq_store:
            await self.faq_store.close()
        if self.vector_store:
            await self.vector_store.close()
        if self.fulltext_store:
            await self.fulltext_store.close()

        print("Done!")

    async def run(self) -> dict[str, Any]:
        """Run the complete ingestion pipeline."""
        print("=" * 60)
        print("Electronics Product Data Ingestion")
        print("=" * 60)

        try:
            # Initialize
            await self.initialize()

            # Load data
            faqs, documents = await self.load_generated_data()

            if not faqs and not documents:
                print("\nNo data to ingest. Please run generate_electronics_data.py first.")
                return {}

            # Ingest
            await self.ingest_faqs(faqs)
            await self.ingest_documents_vector(documents)
            await self.ingest_documents_fulltext(documents)

            # Verify
            stats = await self.verify_ingestion()

            return stats

        finally:
            await self.close()


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest electronics product data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data storage directory",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate data before ingestion",
    )
    parser.add_argument(
        "--faqs",
        type=int,
        default=500,
        help="Number of FAQs to generate",
    )
    parser.add_argument(
        "--docs",
        type=int,
        default=200,
        help="Number of documents to generate",
    )

    args = parser.parse_args()

    # Generate data if requested
    if args.generate:
        print("Generating data...")
        import subprocess

        generate_script = script_dir / "generate_electronics_data.py"
        result = subprocess.run(
            [
                sys.executable,
                str(generate_script),
                "--faqs",
                str(args.faqs),
                "--docs",
                str(args.docs),
            ],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"Generation failed: {result.stderr}")
            return

    # Run ingestion
    pipeline = DataIngestionPipeline(
        data_dir=args.data_dir,
        faq_count=args.faqs,
        doc_count=args.docs,
    )

    stats = await pipeline.run()

    print("\n" + "=" * 60)
    print("Ingestion Summary")
    print("=" * 60)
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())