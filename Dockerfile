# Single-container deployment (Hugging Face Spaces, Docker SDK -- or any
# host that runs a Dockerfile) that serves BOTH the built frontend and the
# FastAPI backend from one process, so there's no separate frontend host,
# no cross-origin request, and nothing extra to configure beyond this file.
#
# Stage 1: build the frontend into static files (frontend/dist).
# Stage 2: install the exact Python environment this project has actually
# been developed and tested against (requirements-lock.txt, not
# requirements.txt's loose ranges -- see that file's own header comment),
# copy in the backend code + the small trained checkpoint, copy the built
# frontend from stage 1, and run uvicorn.

FROM node:22-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS backend
WORKDIR /app

# System deps scapy/cicflowmeter need for packet parsing (libpcap), plus
# build tools some Python packages need to compile from source on a fresh
# platform (torch/torch_geometric ship prebuilt wheels for common
# platforms, but this keeps the build resilient if a wheel isn't available
# for the exact target architecture).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

# Backend code and the real, trained artifacts it needs at inference time.
# Deliberately NOT copying data/ (raw datasets, gitignored, not needed to
# serve predictions) or old-frontend/ (superseded, unused).
COPY api/ ./api/
COPY baseline/ ./baseline/
COPY benchmark/ ./benchmark/
COPY configs/ ./configs/
COPY containment/ ./containment/
COPY counterfactual/ ./counterfactual/
COPY explain/ ./explain/
COPY features/ ./features/
COPY fingerprint/ ./fingerprint/
COPY futures/ ./futures/
COPY graph/ ./graph/
COPY models/ ./models/
COPY momentum/ ./momentum/
COPY rollout/ ./rollout/
COPY checkpoints/ ./checkpoints/

# The built frontend, served as static files by FastAPI itself (see
# api/main.py's StaticFiles mount) -- one origin, no CORS complexity.
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Hugging Face Spaces' Docker SDK expects the app to listen on 7860.
EXPOSE 7860
ENV PORT=7860

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
