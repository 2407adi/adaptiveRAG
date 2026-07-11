# ───────────────────────── STAGE 1: the Node workshop ─────────────────────────
# Disposable: exists only to build web/dist, then gets left behind.

FROM node:22-alpine AS webbuilder
# Rent a tiny pre-fabbed workshop with Node 22 installed.
# "AS webbuilder" nails a name plate on the door — stage 2 will say
# "go fetch the finished goods from the room called webbuilder".

WORKDIR /web
# All following commands happen inside /web (mkdir + cd in one move).

COPY web/package.json web/package-lock.json ./
# Bring in ONLY the parts list first (layer-cache trick: as long as the
# parts list is unchanged, the expensive install below stays cached).

RUN npm ci
# Buy the exact parts the lockfile names — no substitutions.

COPY web/ ./
# NOW bring in the actual source code (changes often; sits below the
# cached install so edits never re-trigger it).
# .dockerignore already keeps node_modules and stale dist out of this copy.

RUN npm run build
# The carpentry: tsc + vite build → finished storefront lands in /web/dist.

# ───────────────────────── STAGE 2: the agency crate ─────────────────────────
# The real, shippable building. Slim Debian + Python 3.11 foundation.

FROM python:3.11-slim
# New empty crate — nothing from the workshop carries over unless we
# explicitly fetch it (that's the whole point of multi-stage).

WORKDIR /app
# The agency's floor. All relative paths in the code ("data/...",
# "config/default.yaml") resolve from here.

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/data/models
# PYTHONUNBUFFERED: print() goes straight to docker logs, no held-back ink.
# HF_HOME: HuggingFace's model closet → pointed INTO data/, which lego 5
# turns into the storage unit — the 90 MB reranker download happens once
# and survives every rebuild, instead of re-downloading per container.

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# OS-level furniture: OCR engine, PDF renderer, faiss's math helper.
# Catalog trash out on the same line so it isn't sealed into the layer.

COPY pyproject.toml ./
RUN mkdir -p src && pip install --no-cache-dir .
# PHASE 1 — the packing list alone, with an empty src/: installs all 20
# dependencies (the slow ~2 GB layer). Stays cached until pyproject changes.

COPY config/ config/
COPY src/ src/
# NOW the real goods: staffing plan + codebase. Changes often — deliberately
# below the dependency layer so edits never re-trigger it.

RUN pip install --no-cache-dir --no-deps -e .
# PHASE 2 — editable install: a signpost in site-packages pointing at
# /app/src. Keeps __file__ under /app so main.py's "3 folders up" lands
# on /app and finds web/dist. --no-deps: parts already bought in phase 1.

COPY --from=webbuilder /web/dist ./web/dist
# Rob the workshop: carry ONLY the finished storefront across.
# Node, npm, node_modules — all left behind for demolition.

EXPOSE 8000
# A label on the crate: "reception is at door 8000." Documentation only —
# the actual door-mapping happens in compose.

CMD ["uvicorn", "adaptiverag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
# The note taped inside the lid: what to run when the crate is opened.
# 0.0.0.0 = listen at the door, not just inside the room.