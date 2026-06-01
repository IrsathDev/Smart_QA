import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from ..database import SessionLocal, get_db
from ..dependencies import current_user
from ..models import Conversation, ConversationDocument, Document, Message, User, now
from ..providers import chat_provider
from ..rag import retrieve
from ..schemas import (
    BranchSelectIn,
    Citation,
    ConversationCreate,
    ConversationDetail,
    ConversationOut,
    ConversationPatch,
    EditMessageIn,
    MessageOut,
    StreamIn,
)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def owned_conversation(db: Session, owner_id: str, conversation_id: str) -> Conversation:
    conversation = db.scalar(select(Conversation).where(Conversation.id == conversation_id, Conversation.owner_id == owner_id))
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


def message_out(message: Message) -> MessageOut:
    return MessageOut(
        id=message.id,
        conversation_id=message.conversation_id,
        parent_message_id=message.parent_message_id,
        role=message.role,
        content=message.content,
        citations=json.loads(message.citations_json),
        created_at=message.created_at,
    )


def active_path(db: Session, conversation: Conversation, leaf_id: str | None = None) -> list[Message]:
    current_id = leaf_id or conversation.active_leaf_id
    path: list[Message] = []
    while current_id:
        message = db.get(Message, current_id)
        if not message or message.conversation_id != conversation.id:
            break
        path.append(message)
        current_id = message.parent_message_id
    return list(reversed(path))


def conversation_detail(db: Session, conversation: Conversation) -> ConversationDetail:
    document_ids = list(db.scalars(select(ConversationDocument.document_id).where(ConversationDocument.conversation_id == conversation.id)))
    return ConversationDetail(
        **ConversationOut.model_validate(conversation).model_dump(),
        messages=[message_out(message) for message in active_path(db, conversation)],
        document_ids=document_ids,
    )


@router.get("", response_model=list[ConversationOut])
def list_conversations(user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[Conversation]:
    return list(db.scalars(select(Conversation).where(Conversation.owner_id == user.id).order_by(Conversation.updated_at.desc())))


@router.post("", response_model=ConversationOut, status_code=201)
def create_conversation(payload: ConversationCreate, user: User = Depends(current_user), db: Session = Depends(get_db)) -> Conversation:
    conversation = Conversation(owner_id=user.id, title=payload.title)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


@router.get("/search", response_model=list[ConversationOut])
def search_conversations(q: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[Conversation]:
    pattern = f"%{q}%"
    ids = select(Message.conversation_id).where(Message.content.ilike(pattern))
    return list(db.scalars(select(Conversation).where(
        Conversation.owner_id == user.id,
        or_(Conversation.title.ilike(pattern), Conversation.id.in_(ids)),
    ).order_by(Conversation.updated_at.desc())))


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> ConversationDetail:
    return conversation_detail(db, owned_conversation(db, user.id, conversation_id))


@router.get("/{conversation_id}/branches", response_model=list[MessageOut])
def list_branches(conversation_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[MessageOut]:
    conversation = owned_conversation(db, user.id, conversation_id)
    parent_ids = select(Message.parent_message_id).where(
        Message.conversation_id == conversation.id,
        Message.parent_message_id.is_not(None),
    )
    leaves = db.scalars(select(Message).where(
        Message.conversation_id == conversation.id,
        Message.role == "assistant",
        Message.id.not_in(parent_ids),
    ).order_by(Message.created_at))
    return [message_out(message) for message in leaves]


@router.patch("/{conversation_id}", response_model=ConversationOut)
def rename_conversation(conversation_id: str, payload: ConversationPatch, user: User = Depends(current_user), db: Session = Depends(get_db)) -> Conversation:
    conversation = owned_conversation(db, user.id, conversation_id)
    conversation.title = payload.title
    conversation.updated_at = now()
    db.commit()
    return conversation


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> None:
    db.delete(owned_conversation(db, user.id, conversation_id))
    db.commit()


@router.post("/{conversation_id}/messages:stream")
def stream_message(conversation_id: str, payload: StreamIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> StreamingResponse:
    conversation = owned_conversation(db, user.id, conversation_id)
    parent_id = payload.parent_message_id if payload.parent_message_id is not None else conversation.active_leaf_id
    if parent_id:
        parent = db.get(Message, parent_id)
        if not parent or parent.conversation_id != conversation.id:
            raise HTTPException(status_code=400, detail="Invalid message parent")
    user_message = Message(conversation_id=conversation.id, parent_message_id=parent_id, role="user", content=payload.content)
    db.add(user_message)
    db.flush()
    conversation.active_leaf_id = user_message.id
    if conversation.title == "New chat":
        conversation.title = payload.content[:60]
    conversation.updated_at = now()
    db.commit()
    db.refresh(user_message)
    return _stream_assistant(db, conversation, user_message, user.id)


@router.post("/{conversation_id}/messages/{message_id}/edit", response_model=MessageOut)
def edit_message(conversation_id: str, message_id: str, payload: EditMessageIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> MessageOut:
    conversation = owned_conversation(db, user.id, conversation_id)
    original = db.get(Message, message_id)
    if not original or original.conversation_id != conversation.id or original.role != "user":
        raise HTTPException(status_code=404, detail="User message not found")
    edited = Message(conversation_id=conversation.id, parent_message_id=original.parent_message_id, role="user", content=payload.content)
    db.add(edited)
    db.flush()
    conversation.active_leaf_id = edited.id
    conversation.updated_at = now()
    db.commit()
    db.refresh(edited)
    return message_out(edited)


@router.post("/{conversation_id}/messages/{message_id}/edit:stream")
def edit_message_stream(conversation_id: str, message_id: str, payload: EditMessageIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> StreamingResponse:
    conversation = owned_conversation(db, user.id, conversation_id)
    original = db.get(Message, message_id)
    if not original or original.conversation_id != conversation.id or original.role != "user":
        raise HTTPException(status_code=404, detail="User message not found")
    edited = Message(conversation_id=conversation.id, parent_message_id=original.parent_message_id, role="user", content=payload.content)
    db.add(edited)
    db.flush()
    conversation.active_leaf_id = edited.id
    conversation.updated_at = now()
    db.commit()
    db.refresh(edited)
    return _stream_assistant(db, conversation, edited, user.id)


@router.post("/{conversation_id}/messages/{message_id}/regenerate:stream")
def regenerate_message(conversation_id: str, message_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> StreamingResponse:
    conversation = owned_conversation(db, user.id, conversation_id)
    assistant = db.get(Message, message_id)
    if not assistant or assistant.conversation_id != conversation.id or assistant.role != "assistant" or not assistant.parent_message_id:
        raise HTTPException(status_code=404, detail="Assistant message not found")
    user_message = db.get(Message, assistant.parent_message_id)
    return _stream_assistant(db, conversation, user_message, user.id)


@router.post("/{conversation_id}/branch", response_model=ConversationDetail)
def select_branch(conversation_id: str, payload: BranchSelectIn, user: User = Depends(current_user), db: Session = Depends(get_db)) -> ConversationDetail:
    conversation = owned_conversation(db, user.id, conversation_id)
    leaf = db.get(Message, payload.leaf_message_id)
    if not leaf or leaf.conversation_id != conversation.id:
        raise HTTPException(status_code=404, detail="Branch leaf not found")
    conversation.active_leaf_id = leaf.id
    db.commit()
    return conversation_detail(db, conversation)


@router.get("/{conversation_id}/export")
def export_conversation(conversation_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)) -> Response:
    conversation = owned_conversation(db, user.id, conversation_id)
    lines = [f"# {conversation.title}", ""]
    for message in active_path(db, conversation):
        lines.extend([f"## {message.role.title()}", "", message.content, ""])
    return Response("\n".join(lines), media_type="text/markdown", headers={"Content-Disposition": 'attachment; filename="conversation.md"'})


def _stream_assistant(db: Session, conversation: Conversation, user_message: Message, owner_id: str) -> StreamingResponse:
    document_ids = list(db.scalars(select(ConversationDocument.document_id).join(Document).where(
        ConversationDocument.conversation_id == conversation.id,
        Document.status == "ready",
    )))
    citations = retrieve(db, owner_id, document_ids, user_message.content)
    context = "\n\n".join(
        f"[{citation.document_name}{f', page {citation.page}' if citation.page else ''}] {citation.excerpt}"
        for citation in citations
    )
    full_history = active_path(db, conversation, user_message.id)
    if len(full_history) > 16:
        summarized = full_history[:-16]
        conversation.summary = "\n".join(f"{item.role}: {item.content[:180]}" for item in summarized[-12:])
        db.commit()
    history = full_history[-16:]
    system = "You are Smart Q&A, a helpful conversational assistant. Use Markdown where it improves clarity."
    if conversation.summary:
        system += "\n\nEarlier conversation summary:\n" + conversation.summary
    if context:
        system += "\nUse the provided document context when relevant. Ground document claims in the context and cite sources inline.\n\nDocument context:\n" + context
    messages = [{"role": "system", "content": system}] + [{"role": item.role, "content": item.content} for item in history]

    def generate() -> Iterator[str]:
        yield _event("message.started", {"user_message": message_out(user_message).model_dump(mode="json")})
        for citation in citations:
            yield _event("citation", citation.model_dump())
        answer: list[str] = []
        try:
            for token in chat_provider().stream(messages):
                answer.append(token)
                yield _event("response.delta", {"delta": token})
            with SessionLocal() as persist_db:
                current = persist_db.get(Conversation, conversation.id)
                assistant = Message(
                    conversation_id=conversation.id,
                    parent_message_id=user_message.id,
                    role="assistant",
                    content="".join(answer).strip(),
                    citations_json=json.dumps([citation.model_dump() for citation in citations]),
                )
                persist_db.add(assistant)
                persist_db.flush()
                current.active_leaf_id = assistant.id
                current.updated_at = now()
                persist_db.commit()
                persist_db.refresh(assistant)
                yield _event("response.completed", {"message": message_out(assistant).model_dump(mode="json")})
        except Exception as exc:
            yield _event("response.error", {"error": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
