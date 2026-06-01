import re
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .config import get_settings
from .models import Chunk, Document
from .providers import embed_query, embed_texts
from .schemas import Citation


def qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=10)


def ensure_collection(vector_size: int) -> None:
    client = qdrant_client()
    collection = get_settings().qdrant_collection
    if not client.collection_exists(collection):
        client.create_collection(collection, vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE))
        client.create_payload_index(collection, "owner_id", qmodels.PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection, "document_id", qmodels.PayloadSchemaType.KEYWORD)


def index_chunks(db: Session, document: Document, parsed_chunks: list[tuple[int | None, str]]) -> None:
    db.execute(delete(Chunk).where(Chunk.document_id == document.id))
    rows = [
        Chunk(document_id=document.id, owner_id=document.owner_id, ordinal=index, page=page, content=content)
        for index, (page, content) in enumerate(parsed_chunks)
    ]
    db.add_all(rows)
    db.flush()
    vectors = embed_texts([row.content for row in rows])
    if vectors:
        ensure_collection(len(vectors[0]))
        qdrant_client().upsert(
            collection_name=get_settings().qdrant_collection,
            points=[
                qmodels.PointStruct(
                    id=row.id,
                    vector=vector,
                    payload={"owner_id": row.owner_id, "document_id": row.document_id, "chunk_id": row.id},
                )
                for row, vector in zip(rows, vectors)
            ],
        )
    db.commit()


def delete_document_vectors(document_id: str) -> None:
    try:
        qdrant_client().delete(
            collection_name=get_settings().qdrant_collection,
            points_selector=qmodels.FilterSelector(filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_id))]
            )),
        )
    except Exception:
        pass


def retrieve(db: Session, owner_id: str, document_ids: list[str], query: str, limit: int = 5) -> list[Citation]:
    if not document_ids:
        return []
    chunks = list(db.scalars(select(Chunk).where(Chunk.owner_id == owner_id, Chunk.document_id.in_(document_ids))))
    document_names = {doc.id: doc.name for doc in db.scalars(select(Document).where(Document.id.in_(document_ids)))}
    lexical = _rank_lexical(chunks, query, limit * 3)
    vector_ids: list[str] = []
    try:
        result = qdrant_client().query_points(
            collection_name=get_settings().qdrant_collection,
            query=embed_query(query),
            query_filter=qmodels.Filter(must=[
                qmodels.FieldCondition(key="owner_id", match=qmodels.MatchValue(value=owner_id)),
                qmodels.FieldCondition(key="document_id", match=qmodels.MatchAny(any=document_ids)),
            ]),
            limit=limit * 3,
        )
        vector_ids = [str(point.payload["chunk_id"]) for point in result.points]
    except Exception:
        vector_ids = []
    ranked_ids = _rrf([chunk.id for chunk in lexical], vector_ids)
    by_id = {chunk.id: chunk for chunk in chunks}
    selected = [by_id[chunk_id] for chunk_id in ranked_ids if chunk_id in by_id][:limit]
    return [
        Citation(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_name=document_names.get(chunk.document_id, "Document"),
            page=chunk.page,
            excerpt=chunk.content[:320],
        )
        for chunk in selected
    ]


def _rank_lexical(chunks: list[Chunk], query: str, limit: int) -> list[Chunk]:
    terms = set(re.findall(r"\w+", query.lower()))
    return sorted(chunks, key=lambda chunk: sum(chunk.content.lower().count(term) for term in terms), reverse=True)[:limit]


def _rrf(*ranked_lists: list[str], k: int = 60) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] += 1 / (k + rank + 1)
    return [item for item, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)]
