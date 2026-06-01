from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserOut(BaseModel):
    id: str
    email: EmailStr
    model_config = ConfigDict(from_attributes=True)


class AuthIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class AuthOut(BaseModel):
    user: UserOut
    csrf_token: str


class ConversationCreate(BaseModel):
    title: str = Field(default="New chat", min_length=1, max_length=160)


class ConversationPatch(BaseModel):
    title: str = Field(min_length=1, max_length=160)


class ConversationOut(BaseModel):
    id: str
    title: str
    active_leaf_id: str | None
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page: int | None
    excerpt: str


class MessageOut(BaseModel):
    id: str
    conversation_id: str
    parent_message_id: str | None
    role: str
    content: str
    citations: list[Citation] = []
    created_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]
    document_ids: list[str]


class StreamIn(BaseModel):
    content: str = Field(min_length=1, max_length=12000)
    parent_message_id: str | None = None


class EditMessageIn(BaseModel):
    content: str = Field(min_length=1, max_length=12000)


class BranchSelectIn(BaseModel):
    leaf_message_id: str


class DocumentOut(BaseModel):
    id: str
    name: str
    mime_type: str
    size_bytes: int
    status: str
    error: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class AttachDocumentsIn(BaseModel):
    document_ids: list[str]
