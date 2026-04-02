# schemas.py
# Step 4: Pydantic — structured, validated output models
#
# WHY Pydantic?
# Without it, the agent returns a raw string. If the LLM hallucinates,
# returns None, or gives a wrong type, your app crashes silently.
# Pydantic catches bad output BEFORE it reaches the database or UI.

from pydantic import BaseModel, field_validator
from typing import Literal


# ═══════════════════════════════════════════════════════════════════════════════
# RAGResponse — validated output for every agent reply
# ═══════════════════════════════════════════════════════════════════════════════

class RAGResponse(BaseModel):
    answer:  str                          # the LLM's reply text
    sources: list[str]                    # page references extracted from reply
    route:   Literal["rag", "chitchat"]   # which agent branch ran

    # ── Validators ────────────────────────────────────────────────────────────
    # @field_validator runs automatically when you call RAGResponse(...)
    # It catches bad data and either fixes or rejects it.

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_empty(cls, v: str) -> str:
        """Ensure the LLM actually returned something."""
        if not v or not v.strip():
            raise ValueError("answer must not be empty")
        return v.strip()

    @field_validator("sources")
    @classmethod
    def deduplicate_sources(cls, v: list[str]) -> list[str]:
        """Remove duplicate page references (e.g. ['Page 1', 'Page 1'])."""
        return list(dict.fromkeys(v))   # preserves order, removes dupes

    @field_validator("route")
    @classmethod
    def route_must_be_valid(cls, v: str) -> str:
        """Literal already enforces 'rag'|'chitchat' — this just lowercases."""
        return v.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# UploadResponse — validated feedback after document indexing
# ═══════════════════════════════════════════════════════════════════════════════

class UploadResponse(BaseModel):
    success:     bool
    chunk_count: int
    message:     str

    @field_validator("chunk_count")
    @classmethod
    def chunks_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("chunk_count must be > 0")
        return v