"""The agency building: constructed once, staffed once, open to the street."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from adaptiverag.config import load_settings
from adaptiverag.pipeline import wire_pipeline
from adaptiverag.api import routes


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
    app.state.conversations = {}                     # cabinet of clipboards: conversation_id → ConversationMemory
    yield                                            # ---- doors open; serve visitors ----
    # ---- closing: nothing to release yet (Chroma persists itself) ----


app = FastAPI(title="AdaptiveRAG API", lifespan=lifespan)

app.add_middleware(                                  # the "visitors welcome" sign
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],         # Vite dev server only; prod is same-origin
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    # Any detective having a breakdown gets wrapped in a calm, uniform envelope —
    # visitors never see a raw Python traceback.
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})