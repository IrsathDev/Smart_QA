from .database import SessionLocal
from .documents import parse_document
from .models import Document
from .rag import index_chunks


def index_document(document_id: str) -> None:
    db = SessionLocal()
    try:
        document = db.get(Document, document_id)
        if not document:
            return
        document.status = "processing"
        db.commit()
        chunks = parse_document(__import__("pathlib").Path(document.path))
        index_chunks(db, document, chunks)
        document.status = "ready"
        document.error = ""
        db.commit()
    except Exception as exc:
        db.rollback()
        document = db.get(Document, document_id)
        if document:
            document.status = "failed"
            document.error = str(exc)[:1000]
            db.commit()
        raise
    finally:
        db.close()
