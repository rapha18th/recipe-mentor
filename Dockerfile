# Recipe Mentor's hosted interface (recipe_mentor/web/app.py), for Cloud Run.
# CPU-only throughout -- there's no GPU on Cloud Run, and no reason to pull
# CUDA-enabled wheels for a training run sized for a weekend CPU laptop.

FROM python:3.12-slim

# soundfile's wheels bundle libsndfile on most platforms, but installed
# explicitly anyway -- cheap, and removes one class of surprise.
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY recipe_mentor/ ./recipe_mentor/

ENV RECIPE_MENTOR_STORE=firestore
ENV PYTHONUNBUFFERED=1

# Cloud Run injects $PORT; default to 8080 for `docker run` outside Cloud Run.
ENV PORT=8080
CMD exec uvicorn recipe_mentor.web.app:app --host 0.0.0.0 --port ${PORT}
