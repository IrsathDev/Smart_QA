# Smart Q&A V3

Smart Q&A is a private multi-user AI chat workspace with optional document retrieval. The V3 app uses FastAPI, PostgreSQL, Qdrant, Redis/RQ, and a Vite React TypeScript client.

## Local Development

1. Copy `.env.example` to `.env`. Keep `DATABASE_URL=sqlite:///./smartqa_v3.db` for a lightweight local database.
2. Add your cloud `REDIS_URL`, `QDRANT_URL`, and `QDRANT_API_KEY`. Cloud Redis URLs may use `rediss://` when TLS is required.
3. Keep `VITE_API_BASE_URL=http://127.0.0.1:8000` so the frontend talks to the API on port `8000` directly.
4. Keep `AI_PROVIDER=mock`, `EMBEDDING_PROVIDER=mock`, `SYNC_INDEXING=true`, and `RATE_LIMITING_ENABLED=false` for the first local smoke test.
5. Install backend packages with `python -m pip install -r requirements.txt`.
6. Start the API with `python -m uvicorn backend.app.main:app --reload --port 8000`.
7. In `frontend/`, run `npm install` and `npm run dev`.

Open `http://localhost:5173`. Use `AI_PROVIDER=nvidia` or `AI_PROVIDER=openai` with the matching API key for model-backed chat. Choose the matching embedding provider and model when indexing production documents.

When document indexing should run outside the API process, set `SYNC_INDEXING=false` and start `python -m backend.worker` in a second terminal. The worker will use your configured cloud Redis queue and Qdrant cluster.

## Later: Docker And Production

Docker is intentionally deferred while the application is under local development. When the product is ready for deployment, build the multi-stage image with `docker build -t smartqa-v3 .`, run one web service and one worker service, attach a persistent volume at `/app/storage`, and configure managed PostgreSQL, Redis, and Qdrant URLs.

## Legacy Import

The former Flask uploads and optional SQLite chat database stay ignored locally so they can be imported once:

```powershell
python -m backend.migrate_legacy --email admin@example.com --password "choose-a-strong-password"
```

This creates the admin account, imports SQLite chat history into `Legacy Import`, copies existing uploads into the admin library, and re-indexes them.

## Database Migrations

The app initializes tables automatically for development. For production schema changes, generate and apply Alembic revisions:

```powershell
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```
