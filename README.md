# 🏷️ Issue Classifier

A web app that classifies software issue reports (bug trackers like GitHub Issues or Jira) as **bug**, **enhancement** or **question** from their title and description. Users can correct a wrong prediction; corrections are stored so they can be used for retraining.

## How it works

```
title + body ──► preprocessing ────► TF-IDF ───────► logistic ────► question ────► label + probabilities
                 (language check,    (word 1–2-      regression     threshold               │
                  links removed,      grams, 100k    (balanced)     (tuned: 0.50)           ▼
                  "?" → qmark,        features)                                      predictions DB
                  question words                                                            ▲
                  and negations kept,                         user confirms / corrects ─────┘
                  lemmatized)
```

- **`train.py`**: loads the dataset, preprocesses it, trains the model and saves the TF-IDF vectorizer, the classifier and the tuned threshold together in `artifacts/model.joblib`, along with `artifacts/metrics.json`.

- **`app.py`**: a Flask API plus a small HTML/JS frontend. Every prediction is stored with a UUID, and `/api/correct` stores the user's corrected label.

## Data

The model is trained on a 72k-issue sample of the **[NLBSE'22 Tool Competition](https://nlbse2022.github.io/tools/) issue report classification dataset**. That dataset contains 800k+ closed GitHub issues from public repositories, labeled bug / enhancement / question, created 2007–2021.

| Label       | Issues | Share |
| ----------- | -----: | ----: |
| bug         | 36,110 | 50.0% |
| enhancement | 29,937 | 41.4% |
| question    |  6,242 |  8.6% |

The dataset is not included in this repository. Place it at `datasets/sample1.csv`, with the columns `issue_label`, `issue_title` and `issue_body`.

## Results

**Final model:** TF-IDF (word 1–2-grams, 100k features) + logistic regression with balanced
class weights, and a tuned decision threshold for `question`.

After removing duplicates and conflicting labels (1,658 issues) and non-English or empty issues
(5,783), 64,848 issues remain, split 70/15/15 (stratified): 45,326 train, 9,727 validation,
9,795 test. Every decision was made on the validation set; the test set was evaluated once, at the end.

| Test set      | Precision | Recall |        F1 |
| ------------- | --------: | -----: | --------: |
| bug           |     0.851 |  0.819 |     0.835 |
| enhancement   |     0.797 |  0.822 |     0.809 |
| question      |     0.503 |  0.541 |     0.521 |
| **macro avg** |     0.717 |  0.727 | **0.722** |

Accuracy 0.795. Validation macro-F1 was 0.724, so the score holds on unseen data.
Classes are imbalanced (question is 9%), so **macro-F1 is the headline metric**.

### How the model got here (validation set)

| Step                                                                       | Macro-F1 | Question F1 |
| -------------------------------------------------------------------------- | -------: | ----------: |
| Majority-class baseline                                                    |    0.226 |       0.000 |
| Random Forest, TF-IDF unigrams (first version)                             |    0.627 |       0.257 |
| Logistic regression, same features                                         |    0.661 |       0.415 |
| + keep question words, negations and `?` in preprocessing                  |    0.691 |       0.476 |
| + word bigrams, 100k features                                              |    0.716 |       0.517 |
| + tuned question threshold (5-fold out-of-fold on train)                   |    0.723 |       0.525 |
| + count `?` only after words (fixes stack traces, found in error analysis) |    0.724 |       0.532 |

Differences below ~0.005 are within run-to-run noise. All runs are tracked with MLflow
(`mlflow ui --port 5001`); the notes are in [`model_comparison.md`](model_comparison.md),
the analysis in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) and
[`notebooks/02_error_analysis.ipynb`](notebooks/02_error_analysis.ipynb).

## Design decisions

- **Deduplicate before splitting.** 1,124 extra copies (mostly issue templates) and 112 texts with
  conflicting labels were found; without dedup, the same issue could be in train and test.
- **Decide on validation, test once.** Experiments never touch the test set or the saved model;
  only `train.py --final` does.
- **Linear model over Random Forest.** TF-IDF is 99.7% zeros; a linear model adds up many weak word
  signals, a tree looks at one word per split. LogReg was better and 3× faster.
- **Keep question signals.** Standard stopword lists delete "how", "why", "not" and `?`, which are
  exactly what separates questions from bug reports.
- **Tune the threshold without touching validation.** Chosen on out-of-fold probabilities from
  5-fold cross-validation on train. The same `predict_with_threshold` function is used in training
  and in the API, so both decide identically.
- **One artifact.** Vectorizer, classifier and threshold are saved together, so they can't get out of sync.
- **Reproducible experiments.** Preprocessing is cached under a hash of the data _and_ the code;
  every run logs its preprocessing version, settings and metrics to MLflow.

## Limitations

- **Label noise.** In the error analysis, 12 of 20 confident mistakes and 9 of 20 missed questions
  turned out to be mislabeled: the model was right. Scores, especially for `question`, underestimate
  the real quality.
- **Bug vs. question is partly undecidable.** Support requests ("I get this error, please help")
  are labeled either way in the data.
- **Repository shortcuts.** Words like `azure` and `msal` are strong question features, because
  some projects use the label for support. This won't transfer to other repositories.
- **Author role is not used.** Issues by outside users are questions far more often (19% vs. 2–4%),
  but the API only receives title and body.
- **Language filter is too strict.** langdetect runs on raw text and rejects some English issues
  that contain logs or URLs (~8% of issues are filtered).
- **About 20 near-duplicates per split remain** after normalized-text deduplication.
- **Sentence embeddings were not evaluated.** The error analysis suggests a small expected gain,
  and shipping them would add PyTorch (~1 GB image) to the service.

## Quickstart

```bash
make install          # create .venv and install dependencies
make train            # train the final model and save it to artifacts/ (needs the dataset, see above)
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

Python, pandas, scikit-learn, NLTK, langdetect, MLflow, Flask, SQLAlchemy, pytest, GitHub Actions.

## Roadmap

- [x] Baselines and model comparison (majority class, Random Forest, linear models)
- [x] Improve recall on `question` (0.15 → 0.54 on test)
- [ ] Docker image and public deployment
- [ ] Retraining pipeline that uses stored user corrections
- [ ] Monitoring of live predictions
