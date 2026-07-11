"""The agency building: constructed once, staffed once, open to the street."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from adaptiverag.config import load_settings
from adaptiverag.pipeline import wire_pipeline
from adaptiverag.api import routes
from adaptiverag.api.auth import RateLimiter
from adaptiverag.api.store import ConversationStore   # the ledger cabinet


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- morning: runs ONCE when the server starts ----
    settings = load_settings()                       # read the staffing plan
    app.state.pipeline = wire_pipeline(              # hire the whole staff (~30s)
        settings,
        collection_name="adaptiverag",
        persist_directory="data/chroma",
    )
    app.state.settings = settings
    # 4.3a: the whiteboard (in-RAM dict, wiped nightly) is retired. The ledger
    # cabinet lives in data/ — same room as Chroma, same Docker volume later.
    app.state.store = ConversationStore("data/conversations.db")
    app.state.api_keys = settings.auth.keys          # the card register (from .env, via load_settings)
    app.state.rate_limiter = RateLimiter(settings.auth.rate_limit_per_minute)   # one shared tally counter
    yield                                            # ---- doors open; serve visitors ----
    # ---- closing: nothing to release yet (Chroma persists itself) ----


app = FastAPI(title="AdaptiveRAG API", lifespan=lifespan)

app.add_middleware(                                  # the "visitors welcome" sign
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],         # Vite dev server only; prod is same-origin
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.public)                    # /health — no doorman (Azure probes)
app.include_router(routes.router)                    # everything else — doorman on duty

# 4.3b: the React storefront. `npm run build` in web/ produces web/dist; FastAPI
# serves it as static files at / — one container, one deploy. API routes were
# registered ABOVE, so they win; everything else falls through to the SPA.
# html=True makes "/" serve index.html. Missing dist (backend-only dev) = no mount.
_web_dist = Path(__file__).resolve().parents[3] / "web" / "dist"
if _web_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_web_dist), html=True), name="web")


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    # Any detective having a breakdown gets wrapped in a calm, uniform envelope —
    # visitors never see a raw Python traceback.
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})