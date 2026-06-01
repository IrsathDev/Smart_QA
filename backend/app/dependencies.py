from fastapi import Depends, Request
from sqlalchemy.orm import Session

from .database import get_db
from .models import User
from .security import read_session, require_csrf


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    user, session = read_session(request, db)
    require_csrf(request, session)
    return user
