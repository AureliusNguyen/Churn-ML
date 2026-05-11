---
title: Churn Report API
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Churn Report API

FastAPI service that powers the Churn Report Next.js frontend. Loads ten
trained scikit-learn / XGBoost models plus a SHAP explainer at startup,
serves predictions and per-feature attributions, and proxies a Groq LLM
for natural-language explanations and retention-email drafts.

## Endpoints

- `GET  /health`                -- liveness + count of models loaded
- `GET  /customers?q=&limit=`   -- search by surname or CustomerId
- `GET  /customers/{id}`        -- full row from churn.csv
- `POST /predict`               -- basic + advanced ensemble probabilities
- `POST /shap`                  -- per-feature contributions for the prediction
- `POST /explain`               -- 3-sentence Groq explanation
- `POST /email`                 -- Groq-drafted retention email

## Local development

```
pip install -r requirements.txt
cp .env.example .env  # fill in GROQ_API_KEY
uvicorn serve:app --reload --port 8000
```

## Deployment (Hugging Face Spaces, Docker SDK)

The README front-matter above declares the Space configuration. Push this
directory to the HF Space's git remote (see `scripts/deploy-hf.sh` at the
repo root) and HF will build the Dockerfile and serve on port 7860.
