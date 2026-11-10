# 1. Base image: small Debian with Python. Must match CI and your .venv.
FROM python:3.11-slim

# 2. Python behavior inside containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app

# 3. Dependencies FIRST, code later (layer cache, see below)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Bake in the NLTK data (instead of downloading on every start)
RUN python -m nltk.downloader -d "$NLTK_DATA" punkt_tab stopwords wordnet

# 5. The model from the GitHub Release
ADD https://github.com/ervinb0/Ticket-Classifier/releases/download/model-20261107-174159/model.joblib artifacts/model.joblib

# 6. The code
COPY app.py .
COPY models/ models/
COPY frontend/ frontend/

# 7. Don't run as root
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# 8. Start the server
EXPOSE 8000
# --timeout 300: with 0.1 CPU the first prediction took ~105 s (measured with --cpus 0.1);
# the default of 30 s would kill the worker mid-request
CMD ["sh", "-c", "python -c 'from models.database import init_db; init_db()' && exec gunicorn --bind 0.0.0.0:${PORT:-8000} --timeout 300 app:app"]
