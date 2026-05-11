"""Feature preparation -- ports prepare_input/prepare_input_2 from main.py.

No `from __future__ import annotations` here: the Pydantic CustomerInput
below is consumed by FastAPI route introspection through slowapi's
wrapper, and string-form annotations confuse the forward-ref resolution.
Real type objects keep this simple.
"""
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field


Location = Literal["France", "Germany", "Spain"]
Gender = Literal["Male", "Female"]


class CustomerInput(BaseModel):
    """Exactly the 10 fields the original Streamlit form collected."""

    credit_score: int = Field(ge=300, le=850)
    location: Location
    gender: Gender
    age: int = Field(ge=18, le=100)
    tenure: int = Field(ge=0, le=50)
    balance: float = Field(ge=0)
    num_products: int = Field(ge=1, le=10)
    has_credit_card: bool
    is_active_member: bool
    estimated_salary: float = Field(ge=0)


def _base_dict(c: CustomerInput) -> dict:
    return {
        "CreditScore": c.credit_score,
        "Age": c.age,
        "Tenure": c.tenure,
        "Balance": c.balance,
        "NumOfProducts": c.num_products,
        "HasCrCard": int(c.has_credit_card),
        "IsActiveMember": int(c.is_active_member),
        "EstimatedSalary": c.estimated_salary,
        "Geography_France": 1 if c.location == "France" else 0,
        "Geography_Germany": 1 if c.location == "Germany" else 0,
        "Geography_Spain": 1 if c.location == "Spain" else 0,
        "Gender_Female": 1 if c.gender == "Female" else 0,
        "Gender_Male": 1 if c.gender == "Male" else 0,
    }


def prepare_basic(c: CustomerInput) -> tuple[pd.DataFrame, dict]:
    d = _base_dict(c)
    return pd.DataFrame([d]), d


def prepare_advanced(c: CustomerInput) -> tuple[pd.DataFrame, dict]:
    d = _base_dict(c)
    d.update({
        "CLV": c.balance * c.estimated_salary / 100_000,
        "TenureAgeRatio": c.tenure / c.age,
        "AgeGroup_MiddleAged": 1 if 30 <= c.age < 50 else 0,
        "AgeGroup_Senior": 1 if 50 <= c.age < 70 else 0,
        "AgeGroup_Elderly": 1 if c.age >= 70 else 0,
    })
    return pd.DataFrame([d]), d
