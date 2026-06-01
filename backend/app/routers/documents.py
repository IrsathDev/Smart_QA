import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from redis import Redis
from rq import Queue
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db
from ..dependencies import current_user
from ..documents import SUPPORTED_EXTENSIONS
from ..models import Conversation, ConversationDocument, Document, User
from ..rag import delete_document_vectors
from ..schemas import AttachDocumentsIn, DocumentOut
from ..tasks import index_document

router = APIRouter(prefix="/api", tags=["documents"])


def owned_conversation(db: Session, owner_id: str, conversation_id: str) -> Conversation:
    conversation = db.scalar(select(Conversation).where(Conversation.id == conversation_id, Conversation.owner_id == owner_id))
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[Document]:
    return list(db.scalars(select(Document).where(Document.owner_id == user.id).order_by(Document.created_at.desc())))


@router.post("/documents", response_model=list[DocumentOut], status_code=status.HTTP_201_CREATED)
def upload_documents(
    files: list[UploadFile] = File(...),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[Document]:
    settings = get_settings()
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Upload at most 10 files at once")
    target_dir = settings.storage_root / user.id
    target_dir.mkdir(parents=True, exist_ok=True)
    documents: list[Document] = []
    for upload in files:
        safe_name = Path(upload.filename or "").name
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {safe_name}")
        path = target_dir / f"{uuid.uuid4()}{suffix}"
        with path.open("wb") as output:
            shutil.copyfileobj(upload.file, output)
        size = path.stat().st_size
        if size > settings.max_upload_mb * 1024 * 1024:
            path.unlink(missing_ok=True)
            raise HTTPException(status_code=413, detail=f"{safe_name} exceeds {settings.max_upload_mb} MB")
        document = Document(owner_id=user.id, name=safe_name, path=str(path), mime_type=upload.content_type or "", size_bytes=size)
        db.add(document)
        db.commit()
        db.refresh(document)
        documents.append(document)
        if settings.sync_indexing:
            index_document(document.id)
            db.refresh(document)
        else:
            Queue("documents", connection=Redis.from_url(settings.redis_url)).enqueue(index_document, document.id)
    return documents


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_document(document_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> None:
    document = db.scalar(select(Document).where(Document.id == document_id, Document.owner_id == user.id))
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    Path(document.path).unlink(missing_ok=True)
    delete_document_vectors(document.id)
    db.delete(document)
    db.commit()


@router.put("/conversations/{conversation_id}/documents", response_model=list[DocumentOut])
def attach_documents(
    conversation_id: str,
    payload: AttachDocumentsIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[Document]:
    owned_conversation(db, user.id, conversation_id)
    documents = list(db.scalars(select(Document).where(Document.owner_id == user.id, Document.id.in_(payload.document_ids))))
    if len(documents) != len(set(payload.document_ids)):
        raise HTTPException(status_code=404, detail="One or more documents were not found")
    db.execute(delete(ConversationDocument).where(ConversationDocument.conversation_id == conversation_id))
    db.add_all([ConversationDocument(conversation_id=conversation_id, document_id=document.id) for document in documents])
    db.commit()
    return documents
