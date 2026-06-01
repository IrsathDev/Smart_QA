import argparse
import shutil
import sqlite3
from pathlib import Path

from sqlalchemy import select

from .app.database import SessionLocal, init_db
from .app.models import Conversation, Document, Message, User
from .app.security import hash_password
from .app.tasks import index_document


def migrate(email: str, password: str, legacy_db: Path, legacy_data: Path) -> None:
    init_db()
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.email == email.lower()))
        if not user:
            user = User(email=email.lower(), password_hash=hash_password(password))
            db.add(user)
            db.flush()
        conversation = Conversation(owner_id=user.id, title="Legacy Import")
        db.add(conversation)
        db.flush()
        parent_id = None
        if legacy_db.exists():
            connection = sqlite3.connect(legacy_db)
            for role, content in connection.execute("SELECT role, content FROM chats ORDER BY id"):
                message = Message(conversation_id=conversation.id, parent_message_id=parent_id, role=role, content=content)
                db.add(message)
                db.flush()
                parent_id = message.id
            connection.close()
        conversation.active_leaf_id = parent_id
        storage = Path("./storage") / user.id
        storage.mkdir(parents=True, exist_ok=True)
        imported: list[Document] = []
        if legacy_data.exists():
            for source in legacy_data.iterdir():
                if not source.is_file():
                    continue
                target = storage / source.name
                shutil.copy2(source, target)
                document = Document(owner_id=user.id, name=source.name, path=str(target), size_bytes=target.stat().st_size)
                db.add(document)
                imported.append(document)
        db.commit()
        for document in imported:
            index_document(document.id)
    print(f"Imported legacy data for {email}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--legacy-db", type=Path, default=Path("./chat.db"))
    parser.add_argument("--legacy-data", type=Path, default=Path("./data"))
    args = parser.parse_args()
    migrate(args.email, args.password, args.legacy_db, args.legacy_data)
