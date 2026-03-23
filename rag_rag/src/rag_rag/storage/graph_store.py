"""
Graph Store Implementation.

Neo4j-based knowledge graph storage for entity and relationship queries.
"""

import asyncio
from typing import Any, Optional

from rag_rag.storage.base import RetrievableStore, StoreStatus
from rag_rag.core.exceptions import GraphStoreError
from rag_rag.core.logging import get_logger

logger = get_logger("rag_rag.storage.graph")


class GraphStore(RetrievableStore):
    """
    Graph Store with Neo4j backend.

    Features:
    - Entity and relationship storage
    - Graph traversal queries
    - Entity extraction support
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ):
        super().__init__("graph_store")
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver: Optional[Any] = None

    async def initialize(self) -> None:
        """Initialize Neo4j connection."""
        try:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )

            # Verify connection
            await self._verify_connection()

            # Create constraints and indexes
            await self._create_constraints()

            self._set_ready()
            logger.info(f"Graph Store initialized: {self.uri}")

        except ImportError:
            self._set_error("neo4j not installed")
            raise GraphStoreError(
                "neo4j not installed. Install with: pip install neo4j"
            )
        except Exception as e:
            self._set_error(str(e))
            raise GraphStoreError(f"Failed to initialize graph store: {e}")

    async def _verify_connection(self) -> None:
        """Verify Neo4j connection."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run("RETURN 1")
            await result.single()

    async def _create_constraints(self) -> None:
        """Create constraints and indexes for the graph schema."""
        constraints = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
        ]

        indexes = [
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)",
        ]

        async with self._driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    await session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint creation skipped: {e}")

            for index in indexes:
                try:
                    await session.run(index)
                except Exception as e:
                    logger.debug(f"Index creation skipped: {e}")

    async def health_check(self) -> bool:
        """Check if graph store is healthy."""
        try:
            if self._driver is None:
                return False
            await self._verify_connection()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the Neo4j connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
        self._set_closed()

    # === Entity Operations ===

    async def create_entity(
        self,
        entity_type: str,
        entity_id: str,
        properties: dict[str, Any],
    ) -> bool:
        """
        Create an entity in the graph.

        Args:
            entity_type: Type of entity (Person, Document, Concept, etc.)
            entity_id: Unique entity ID
            properties: Entity properties

        Returns:
            True if created
        """
        if self._driver is None:
            raise GraphStoreError("Store not initialized")

        # Sanitize entity_type to prevent injection
        entity_type = entity_type.replace("`", "").replace("'", "")

        query = f"""
        MERGE (e:{entity_type} {{id: $entity_id}})
        SET e += $properties
        RETURN e
        """

        async with self._driver.session(database=self.database) as session:
            result = await session.run(
                query,
                entity_id=entity_id,
                properties=properties,
            )
            return await result.single() is not None

    async def create_relation(
        self,
        from_id: str,
        relation_type: str,
        to_id: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Create a relationship between entities.

        Args:
            from_id: Source entity ID
            relation_type: Relationship type (BELONGS_TO, RELATED_TO, etc.)
            to_id: Target entity ID
            properties: Optional relationship properties

        Returns:
            True if created
        """
        if self._driver is None:
            raise GraphStoreError("Store not initialized")

        # Sanitize relation_type
        relation_type = relation_type.replace("`", "").replace("'", "")

        query = f"""
        MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
        MERGE (a)-[r:{relation_type}]->(b)
        SET r += $properties
        RETURN r
        """

        async with self._driver.session(database=self.database) as session:
            result = await session.run(
                query,
                from_id=from_id,
                to_id=to_id,
                properties=properties or {},
            )
            return await result.single() is not None

    async def get_entity(self, entity_id: str) -> Optional[dict[str, Any]]:
        """Get an entity by ID."""
        if self._driver is None:
            raise GraphStoreError("Store not initialized")

        query = """
        MATCH (e {id: $entity_id})
        RETURN e.id as id, labels(e) as types, properties(e) as properties
        """

        async with self._driver.session(database=self.database) as session:
            result = await session.run(query, entity_id=entity_id)
            record = await result.single()

            if record:
                return {
                    "id": record["id"],
                    "types": record["types"],
                    "properties": record["properties"],
                }
            return None

    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships."""
        if self._driver is None:
            raise GraphStoreError("Store not initialized")

        query = """
        MATCH (e {id: $entity_id})
        DETACH DELETE e
        RETURN count(e) as deleted
        """

        async with self._driver.session(database=self.database) as session:
            result = await session.run(query, entity_id=entity_id)
            record = await result.single()
            return record and record["deleted"] > 0

    # === Search Operations ===

    async def search(
        self,
        query: str,
        top_k: int = 10,
        entity_types: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Search for entities by name or property.

        Args:
            query: Search query
            top_k: Maximum results
            entity_types: Optional filter by entity types

        Returns:
            List of matching entities
        """
        if self._driver is None:
            raise GraphStoreError("Store not initialized")

        # Build query based on filters
        if entity_types:
            type_filter = " OR ".join([f"e:{t}" for t in entity_types])
            cypher = f"""
            MATCH (e)
            WHERE ({type_filter}) AND (
                e.name CONTAINS $query OR
                e.content CONTAINS $query
            )
            RETURN e.id as id, labels(e) as types, properties(e) as properties,
                   coalesce(e.name, '') as name
            LIMIT $limit
            """
        else:
            cypher = """
            MATCH (e)
            WHERE e.name CONTAINS $query OR e.content CONTAINS $query
            RETURN e.id as id, labels(e) as types, properties(e) as properties,
                   coalesce(e.name, '') as name
            LIMIT $limit
            """

        async with self._driver.session(database=self.database) as session:
            result = await session.run(cypher, query=query, limit=top_k)
            records = await result.data()

            return [
                {
                    "document_id": record["id"],
                    "content": record.get("properties", {}).get("content", record.get("name", "")),
                    "score": 1.0,  # Neo4j doesn't provide relevance scores directly
                    "source": "graph",
                    "metadata": {
                        "types": record["types"],
                        "properties": record["properties"],
                    },
                }
                for record in records
            ]

    async def find_related(
        self,
        entity_id: str,
        relation_types: Optional[list[str]] = None,
        depth: int = 1,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Find entities related to a given entity.

        Args:
            entity_id: Source entity ID
            relation_types: Optional filter by relation types
            depth: Traversal depth
            limit: Maximum results

        Returns:
            List of related entities
        """
        if self._driver is None:
            raise GraphStoreError("Store not initialized")

        if relation_types:
            rel_filter = "|".join(relation_types)
            cypher = f"""
            MATCH (e {{id: $entity_id}})-[r:{rel_filter}*1..{depth}]-(related)
            RETURN DISTINCT related.id as id, labels(related) as types,
                   properties(related) as properties
            LIMIT $limit
            """
        else:
            cypher = f"""
            MATCH (e {{id: $entity_id}})-[r*1..{depth}]-(related)
            RETURN DISTINCT related.id as id, labels(related) as types,
                   properties(related) as properties
            LIMIT $limit
            """

        async with self._driver.session(database=self.database) as session:
            result = await session.run(
                cypher,
                entity_id=entity_id,
                limit=limit,
            )
            records = await result.data()

            return [
                {
                    "id": record["id"],
                    "types": record["types"],
                    "properties": record["properties"],
                }
                for record in records
            ]

    # === RetrievableStore Interface ===

    async def add(
        self,
        documents: list[dict[str, Any]],
        **kwargs: Any,
    ) -> int:
        """Add documents as Document entities."""
        count = 0
        for doc in documents:
            success = await self.create_entity(
                "Document",
                doc.get("id", ""),
                {
                    "content": doc.get("content", ""),
                    **doc.get("metadata", {}),
                },
            )
            if success:
                count += 1
        return count

    async def delete(self, document_ids: list[str]) -> int:
        """Delete documents by IDs."""
        count = 0
        for doc_id in document_ids:
            if await self.delete_entity(doc_id):
                count += 1
        return count

    async def get(self, document_id: str) -> Optional[dict[str, Any]]:
        """Get a document by ID."""
        entity = await self.get_entity(document_id)
        if entity:
            return {
                "document_id": entity["id"],
                "content": entity["properties"].get("content", ""),
                "metadata": entity["properties"],
            }
        return None

    async def count(self) -> int:
        """Get total entity count."""
        if self._driver is None:
            return 0

        query = "MATCH (e) RETURN count(e) as count"

        async with self._driver.session(database=self.database) as session:
            result = await session.run(query)
            record = await result.single()
            return record["count"] if record else 0

    async def clear(self) -> int:
        """Clear all entities."""
        if self._driver is None:
            return 0

        count = await self.count()

        query = "MATCH (n) DETACH DELETE n"

        async with self._driver.session(database=self.database) as session:
            await session.run(query)

        return count

    def get_info(self) -> dict[str, Any]:
        """Get store information."""
        return {
            "name": self.name,
            "status": self.status.value,
            "uri": self.uri,
            "database": self.database,
        }