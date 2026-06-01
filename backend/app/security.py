import hashlib
import secrets
from datetime import timedelta, timezone

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from fastapi import HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .models import User, UserSession, now

SESSION_COOKIE = "smartqa_session"
CSRF_COOKIE = "smartqa_csrf"
ph = PasswordHasher()


def hash_password(password: str) -> str:
    return ph.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return ph.verify(password_hash, password)
    except VerificationError:
        return False


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(db: Session, user: User, response: Response) -> str:
    raw_token = secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(24)
    db.add(UserSession(
        user_id=user.id,
        token_hash=token_hash(raw_token),
        csrf_token=csrf,
        expires_at=now() + timedelta(days=14),
    ))
    db.commit()
    secure = get_settings().cookie_secure
    response.set_cookie(SESSION_COOKIE, raw_token, httponly=True, secure=secure, samesite="lax", max_age=1209600)
    response.set_cookie(CSRF_COOKIE, csrf, httponly=False, secure=secure, samesite="lax", max_age=1209600)
    return csrf


def read_session(request: Request, db: Session) -> tuple[User, UserSession]:
    raw_token = request.cookies.get(SESSION_COOKIE)
    if not raw_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    session = db.scalar(select(UserSession).where(UserSession.token_hash == token_hash(raw_token)))
    expires_at = session.expires_at if session else None
    if expires_at and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if not session or expires_at < now():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")
    user = db.get(User, session.user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user, session


def require_csrf(request: Request, session: UserSession) -> None:
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    header = request.headers.get("X-CSRF-Token")
    cookie = request.cookies.get(CSRF_COOKIE)
    if not header or not cookie or header != cookie or header != session.csrf_token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token")
