"""FastAPI entrypoint for the Churn Report.

Loads ten trained models + the dataset once at startup, then exposes the
endpoints the Next.js frontend calls.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from features import CustomerInput, prepare_advanced, prepare_basic
from inference import (
    load_all_models,
    predict_advanced,
    predict_basic,
    shap_for_advanced,
)
from llm import explain_prediction, generate_email

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

origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- response models ----------


class CustomerSearchHit(BaseModel):
    customer_id: int
    surname: str
    geography: str
    age: int


class CustomerDetail(BaseModel):
    customer_id: int
    surname: str
    credit_score: int
    geography: str
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


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": state["models_loaded"]}


@app.get("/customers", response_model=list[CustomerSearchHit])
def search_customers(q: str = Query(default="", description="Surname or CustomerId substring"), limit: int = 20):
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
            geography=str(r["Geography"]),
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
        geography=str(r["Geography"]),
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
def predict(customer: CustomerInput):
    basic_df, _ = prepare_basic(customer)
    adv_df, _ = prepare_advanced(customer)
    basic, basic_avg = predict_basic(basic_df)
    advanced, advanced_avg = predict_advanced(adv_df)
    return PredictResponse(
        basic=basic, advanced=advanced, basic_avg=basic_avg, advanced_avg=advanced_avg
    )


@app.post("/shap", response_model=ShapResponse)
def shap_endpoint(customer: CustomerInput):
    adv_df, _ = prepare_advanced(customer)
    result = shap_for_advanced(adv_df)
    return ShapResponse(**result)


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
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
def email(req: EmailRequest):
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
