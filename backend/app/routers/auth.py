from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..dependencies import current_user
from ..models import User, UserSession
from ..schemas import AuthIn, AuthOut, UserOut
from ..security import SESSION_COOKIE, create_session, hash_password, read_session, require_csrf, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=AuthOut, status_code=status.HTTP_201_CREATED)
def register(payload: AuthIn, response: Response, db: Session = Depends(get_db)) -> AuthOut:
    email = payload.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status_code=409, detail="Email is already registered")
    user = User(email=email, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return AuthOut(user=UserOut.model_validate(user), csrf_token=create_session(db, user, response))


@router.post("/login", response_model=AuthOut)
def login(payload: AuthIn, response: Response, db: Session = Depends(get_db)) -> AuthOut:
    user = db.scalar(select(User).where(User.email == payload.email.lower()))
    if not user or not verify_password(user.password_hash, payload.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return AuthOut(user=UserOut.model_validate(user), csrf_token=create_session(db, user, response))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, response: Response, db: Session = Depends(get_db)) -> None:
    _, session = read_session(request, db)
    require_csrf(request, session)
    db.delete(session)
    db.commit()
    response.delete_cookie(SESSION_COOKIE)
    response.delete_cookie("smartqa_csrf")


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)) -> User:
    return user
