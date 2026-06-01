from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from ..database import SessionLocal

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready() -> dict[str, str]:
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database is unavailable") from exc
