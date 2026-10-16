# 🏷️ Issue Classifier

A web app that classifies software issue reports (bug trackers like GitHub Issues or Jira) as **bug**, **enhancement** or **question** from their title and description. Users can correct a wrong prediction; corrections are stored so they can be used for retraining.

## How it works

```
title + body ──► preprocessing ──► TF-IDF ──► Random Forest ──► label + probabilities
                 (language check,                                      │
                  link/noise removal,                                  ▼
                  stopwords, lemmatization)               predictions DB (SQLite)
                                                                       ▲
                                         user confirms / corrects ─────┘
```

- **`train.py`**: loads the dataset, preprocesses it, trains the model and saves the TF-IDF vectorizer and classifier together as **one sklearn `Pipeline`** in `artifacts/model.joblib`, along with `artifacts/metrics.json`.
- **`app.py`**: a Flask API plus a small HTML/JS frontend. Every prediction is stored with a UUID, and `/api/correct` stores the user's corrected label.

## Data

The model is trained on a 72k-issue sample of the **[NLBSE'22 Tool Competition](https://nlbse2022.github.io/tools/) issue report classification dataset**. That dataset contains 800k+ closed GitHub issues from public repositories, labeled bug / enhancement / question, created 2007–2021.

| Label | Issues | Share |
| --- | ---: | ---: |
| bug | 36,110 | 50.0% |
| enhancement | 29,937 | 41.4% |
| question | 6,242 | 8.6% |

The dataset is not included in this repository. Place it at `datasets/sample1.csv`, with the columns `issue_label`, `issue_title` and `issue_body`.

## Results

TF-IDF (10k features) + Random Forest (100 trees, balanced class weights). The data is split 70/15/15 (stratified) after removing non-English and empty issues (66,263 kept). Scores below are on the held-out **test set**:

| Label | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| bug | 0.785 | 0.862 | 0.822 |
| enhancement | 0.778 | 0.812 | 0.795 |
| question | 0.633 | **0.151** | 0.244 |
| **macro avg** | 0.732 | 0.609 | **0.620** |

Accuracy is 0.779, but because the classes are imbalanced, **macro-F1 is the headline metric**.

**Known weakness:** the model rarely predicts `question`. Only 8.6% of training issues are questions, and many of them read like bug reports. Improving this is the next step (see Roadmap). Running `python train.py` writes the full metrics to `artifacts/metrics.json`.

## Quickstart

```bash
make install          # create .venv and install dependencies
make train            # train the model (needs the dataset, see above)
make run              # start the app on http://127.0.0.1:8000
make test             # run the tests (no dataset or model needed)
```

### API

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "App crashes on startup", "body": "After updating to 2.1 the app closes immediately."}'
# -> {"id": "...", "predicted_label": "bug", "confidence": 0.95,
#     "probabilities": {...}, "model_version": "..."}

curl -X POST http://127.0.0.1:8000/api/correct \
  -H "Content-Type: application/json" \
  -d '{"id": "<id from predict>", "corrected_label": "question"}'
```

Configuration is done with environment variables: `PORT` (default 8000), `MODEL_PATH`, `DATABASE_URL` (any SQLAlchemy URL), `FLASK_DEBUG=1`.

## Tech stack

Python, scikit-learn, NLTK, langdetect, Flask, SQLAlchemy, pytest, GitHub Actions.

## Roadmap

- [ ] Compare against baselines (majority class, linear models) and a transformer model
- [ ] Improve recall on `question`
- [ ] Docker image and public deployment
- [ ] Retraining pipeline that uses stored user corrections
