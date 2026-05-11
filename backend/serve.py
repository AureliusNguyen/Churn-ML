"""FastAPI entrypoint for the Churn Report.

Loads ten trained models + the dataset once at startup, then exposes the
endpoints the Next.js frontend calls.

NOTE: deliberately no `from __future__ import annotations` here -- with
slowapi's wrapper around our route handlers, FastAPI's runtime
introspection can't resolve forward-referenced types like CustomerInput
(they look up the wrapper's __globals__ instead of this module's), so the
app fails to import. With this future import removed, the annotations
are real types at decoration time and slowapi composes cleanly.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from features import CustomerInput, prepare_advanced, prepare_basic
from inference import (
    load_all_models,
    predict_advanced,
    predict_basic,
    shap_for_advanced,
)
from llm import explain_prediction, generate_email


def real_ip(request: Request) -> str:
    """Extract the originating client IP, honoring proxy headers.

    HF Spaces front the container with a proxy that sets X-Forwarded-For
    and X-Proxied-Host; without trusting one of those, every request
    looks like it came from the proxy's internal IP and the limiter
    would treat all traffic as a single client.
    """
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


limiter = Limiter(key_func=real_ip)

load_dotenv()

DATA_PATH = Path(__file__).parent / "data" / "churn.csv"

state: dict = {"df": None, "models_loaded": 0, "churned_stats": "", "non_churned_stats": ""}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["df"] = pd.read_csv(DATA_PATH)
    state["models_loaded"] = load_all_models()
    pd.set_option("display.max_colwidth", None)
    state["churned_stats"] = state["df"][state["df"]["Exited"] == 1].describe().to_string()
    state["non_churned_stats"] = state["df"][state["df"]["Exited"] == 0].describe().to_string()
    yield


app = FastAPI(title="Churn Report API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_raw_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
# Defense in depth: a wildcard combined with allow_credentials=True is a
# common footgun -- browsers reject it but FastAPI/Starlette will happily
# echo the request origin instead. Reject wildcards explicitly so a config
# mistake doesn't silently open the API to every origin with credentials.
if "*" in _raw_origins:
    raise RuntimeError(
        "ALLOWED_ORIGINS='*' is unsafe with allow_credentials=True. "
        "List specific origins instead."
    )
print(f"[cors] allowed origins: {_raw_origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_raw_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ---------- response models ----------


class CustomerSearchHit(BaseModel):
    customer_id: int
    surname: str
    location: str
    age: int


class CustomerDetail(BaseModel):
    customer_id: int
    surname: str
    credit_score: int
    location: str
    gender: str
    age: int
    tenure: int
    balance: float
    num_products: int
    has_credit_card: bool
    is_active_member: bool
    estimated_salary: float
    exited: bool


class PredictResponse(BaseModel):
    basic: dict[str, float]
    advanced: dict[str, float]
    basic_avg: float
    advanced_avg: float


class ShapItem(BaseModel):
    feature: str
    value: float
    contribution: float


class ShapResponse(BaseModel):
    base_value: float
    expected_prob: float
    predicted_prob: float
    shap_values: list[ShapItem]


class ExplainRequest(BaseModel):
    customer: CustomerInput
    surname: str
    probability: float


class ExplainResponse(BaseModel):
    text: str


class EmailRequest(BaseModel):
    customer: CustomerInput
    surname: str
    probability: float
    explanation: str


# ---------- endpoints ----------


@app.get("/", include_in_schema=False)
def root():
    """Land any visitor (or HF's internal monitor) on the auto Swagger UI
    instead of a 404 for /."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs", status_code=307)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": state["models_loaded"]}


@app.get("/customers", response_model=list[CustomerSearchHit])
def search_customers(
    q: str = Query(default="", description="Surname or CustomerId substring"),
    limit: int = Query(default=20, ge=1, le=100),
):
    df = state["df"]
    if df is None:
        raise HTTPException(503, "Dataset not loaded yet.")
    if not q:
        rows = df.head(limit)
    else:
        ql = q.lower()
        mask = (
            df["Surname"].str.lower().str.contains(ql, na=False)
            | df["CustomerId"].astype(str).str.contains(q, na=False)
        )
        rows = df[mask].head(limit)
    return [
        CustomerSearchHit(
            customer_id=int(r["CustomerId"]),
            surname=str(r["Surname"]),
            location=str(r["Geography"]),
            age=int(r["Age"]),
        )
        for _, r in rows.iterrows()
    ]


@app.get("/customers/{customer_id}", response_model=CustomerDetail)
def get_customer(customer_id: int):
    df = state["df"]
    if df is None:
        raise HTTPException(503, "Dataset not loaded yet.")
    matches = df[df["CustomerId"] == customer_id]
    if matches.empty:
        raise HTTPException(404, f"Customer {customer_id} not found.")
    r = matches.iloc[0]
    return CustomerDetail(
        customer_id=int(r["CustomerId"]),
        surname=str(r["Surname"]),
        credit_score=int(r["CreditScore"]),
        location=str(r["Geography"]),
        gender=str(r["Gender"]),
        age=int(r["Age"]),
        tenure=int(r["Tenure"]),
        balance=float(r["Balance"]),
        num_products=int(r["NumOfProducts"]),
        has_credit_card=bool(r["HasCrCard"]),
        is_active_member=bool(r["IsActiveMember"]),
        estimated_salary=float(r["EstimatedSalary"]),
        exited=bool(r["Exited"]),
    )


@app.post("/predict", response_model=PredictResponse)
@limiter.limit("60/minute")
def predict(request: Request, customer: CustomerInput):
    basic_df, _ = prepare_basic(customer)
    adv_df, _ = prepare_advanced(customer)
    basic, basic_avg = predict_basic(basic_df)
    advanced, advanced_avg = predict_advanced(adv_df)
    return PredictResponse(
        basic=basic, advanced=advanced, basic_avg=basic_avg, advanced_avg=advanced_avg
    )


@app.post("/shap", response_model=ShapResponse)
@limiter.limit("60/minute")
def shap_endpoint(request: Request, customer: CustomerInput):
    adv_df, _ = prepare_advanced(customer)
    result = shap_for_advanced(adv_df)
    return ShapResponse(**result)


@app.post("/explain", response_model=ExplainResponse)
@limiter.limit("10/minute")
def explain(request: Request, req: ExplainRequest):
    _, input_dict = prepare_advanced(req.customer)
    text = explain_prediction(
        req.probability,
        input_dict,
        req.surname,
        state["churned_stats"],
        state["non_churned_stats"],
    )
    return ExplainResponse(text=text)


@app.post("/email", response_model=ExplainResponse)
@limiter.limit("10/minute")
def email(request: Request, req: EmailRequest):
    _, input_dict = prepare_advanced(req.customer)
    text = generate_email(req.probability, input_dict, req.explanation, req.surname)
    return ExplainResponse(text=text)


@app.get("/dataset/summary")
def dataset_summary():
    """Tiny aggregates for the editorial ticker."""
    df = state["df"]
    if df is None:
        raise HTTPException(503, "Dataset not loaded yet.")
    return {
        "total": int(len(df)),
        "churn_rate": float(df["Exited"].mean()),
        "mean_tenure": float(df["Tenure"].mean()),
        "mean_balance": float(df["Balance"].mean()),
        "mean_credit_score": float(df["CreditScore"].mean()),
        "geographies": df["Geography"].value_counts().to_dict(),
    }
