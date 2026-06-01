from contextlib import asynccontextmanager
from pathlib import Path
import logging
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from redis import Redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from .config import get_settings
from .database import init_db
from .routers import auth, conversations, documents, health
from .providers import chat_provider

settings = get_settings()
logger = logging.getLogger("uvicorn.error")


def _allowed_origins(frontend_origin: str) -> list[str]:
    origins = {frontend_origin.rstrip("/")}
    parts = urlsplit(frontend_origin)
    if parts.hostname in {"localhost", "127.0.0.1"}:
        alternate_host = "127.0.0.1" if parts.hostname == "localhost" else "localhost"
        alternate = urlunsplit((parts.scheme, f"{alternate_host}:{parts.port}" if parts.port else alternate_host, parts.path, parts.query, parts.fragment))
        origins.add(alternate.rstrip("/"))
    return sorted(origins)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    if settings.app_env == "production" and settings.secret_key == "dev-secret-change-me":
        raise RuntimeError("SECRET_KEY must be configured in production")
    init_db()
    logger.info(
        "Smart Q&A startup: ai_provider=%s resolved_provider=%s embedding_provider=%s nvidia_key_set=%s frontend_origin=%s rate_limiting=%s",
        settings.ai_provider,
        chat_provider().__class__.__name__,
        settings.embedding_provider,
        bool(settings.nvidia_api_key),
        settings.frontend_origin,
        settings.rate_limiting_enabled,
    )
    yield


app = FastAPI(title="Smart Q&A V3", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(settings.frontend_origin),
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth.router)
app.include_router(conversations.router)
app.include_router(documents.router)
app.include_router(health.router)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if not settings.rate_limiting_enabled:
        return await call_next(request)

    limits = {
        "/api/auth/login": (15, 60),
        "/api/auth/register": (10, 3600),
    }
    limit, window = limits.get(request.url.path, (60, 60) if request.url.path.endswith("messages:stream") else (0, 0))
    if limit and settings.app_env != "test":
        try:
            client = Redis.from_url(
                settings.redis_url,
                socket_connect_timeout=0.25,
                socket_timeout=0.25,
                retry=Retry(NoBackoff(), 0),
            )
            key = f"rate:{request.client.host if request.client else 'unknown'}:{request.url.path}"
            count = client.incr(key)
            if count == 1:
                client.expire(key, window)
            if count > limit:
                return JSONResponse({"detail": "Too many requests"}, status_code=429)
        except Exception:
            pass
    return await call_next(request)

frontend_dist = Path(__file__).parents[2] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
