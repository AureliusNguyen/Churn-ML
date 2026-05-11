"""Model loading + inference + SHAP explainer."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

MODELS_DIR = Path(__file__).parent / "models"

_models: dict[str, Any] = {}
_shap_explainer: shap.TreeExplainer | None = None


def _load(name: str) -> Any:
    with open(MODELS_DIR / name, "rb") as f:
        return pickle.load(f)


def load_all_models() -> int:
    """Load every .pkl file once at app startup. Returns count loaded."""
    global _shap_explainer
    files = [
        "dt_model.pkl",
        "knn_model.pkl",
        "nb_model.pkl",
        "rf_model.pkl",
        "svm_model.pkl",
        "voting_clf.pkl",
        "xgb_model.pkl",
        "xgboost_feature_engineered.pkl",
        "xgboost_SMOTE.pkl",
        "stacking_model.pkl",
        "best_xgb_clf.pkl",
    ]
    for fname in files:
        key = fname.replace(".pkl", "")
        _models[key] = _load(fname)

    _shap_explainer = shap.TreeExplainer(_models["best_xgb_clf"])
    return len(_models)


def predict_basic(df: pd.DataFrame) -> tuple[dict[str, float], float]:
    probs = {
        "XGBoost": float(_models["xgb_model"].predict_proba(df)[0][1]),
        "Random Forest": float(_models["rf_model"].predict_proba(df)[0][1]),
        "K-Nearest Neighbor": float(_models["knn_model"].predict_proba(df)[0][1]),
        "Support Vector Machine": float(_models["svm_model"].predict_proba(df)[0][1]),
    }
    return probs, float(np.mean(list(probs.values())))


def predict_advanced(df: pd.DataFrame) -> tuple[dict[str, float], float]:
    probs = {
        "XGBoost with Feature Engineering": float(
            _models["xgboost_feature_engineered"].predict_proba(df)[0][1]
        ),
        "XGBoost with SMOTE": float(_models["xgboost_SMOTE"].predict_proba(df)[0][1]),
        "Best XGB": float(_models["best_xgb_clf"].predict_proba(df)[0][1]),
        "Stacking": float(_models["stacking_model"].predict_proba(df)[0][1]),
    }
    return probs, float(np.mean(list(probs.values())))


def shap_for_advanced(df: pd.DataFrame) -> dict:
    """SHAP attribution for one row against best_xgb_clf (18 features)."""
    if _shap_explainer is None:
        raise RuntimeError("SHAP explainer not initialized; call load_all_models first.")

    raw = _shap_explainer(df)
    values = np.asarray(raw.values).reshape(-1)
    base_value = float(np.asarray(raw.base_values).reshape(-1)[0])
    feature_names = list(df.columns)
    feature_values = df.iloc[0].tolist()

    contributions = [
        {
            "feature": name,
            "value": float(val) if not isinstance(val, (int, float)) else val,
            "contribution": float(contrib),
        }
        for name, val, contrib in zip(feature_names, feature_values, values)
    ]

    predicted_prob = float(_models["best_xgb_clf"].predict_proba(df)[0][1])
    expected_prob = float(1.0 / (1.0 + np.exp(-base_value)))

    return {
        "base_value": base_value,
        "expected_prob": expected_prob,
        "predicted_prob": predicted_prob,
        "shap_values": contributions,
    }
