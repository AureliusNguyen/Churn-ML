"""Groq-backed natural language explanations and retention emails.

Ports explain_prediction() and generate_email() from the original main.py,
keeping the same model and the same prompt structure so output quality is
unchanged.
"""
from __future__ import annotations

import os

from groq import Groq

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return Groq(api_key=api_key)


def explain_prediction(
    probability: float,
    input_dict: dict,
    surname: str,
    churned_stats: str,
    non_churned_stats: str,
) -> str:
    pct = round(probability * 100, 1)
    prompt = f"""
  You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models.
  Your machine learning model has predicted that a customer named {surname} will churn with a probability of {pct}% based on the information provided below.
  Here is the information about the customer:
  {input_dict}

  Here are the machine learning model's top 10 features that are most important and should be used to make the churn prediction:
  +---------------------+-------------+
  |      Feature        | Importance  |
  +---------------------+-------------+
  | NumOfProducts       |   0.323888  |
  | IsActiveMember      |   0.164146  |
  | Age                 |   0.109550  |
  | Geography_Germany   |   0.091373  |
  | Balance             |   0.052786  |
  | Geography_France    |   0.046463  |
  | Gender_Female       |   0.045283  |
  | Geography_Spain     |   0.036855  |
  | CreditScore         |   0.035005  |
  | EstimatedSalary     |   0.032655  |
  | HasCrCard           |   0.031940  |
  | Tenure              |   0.030054  |
  | Gender_Male         |   0.000000  |
  +---------------------+-------------+

  Here are the summary statistics for the churned customers:
  {churned_stats}

  Here are the summary statistics for the non-churned customers:
  {non_churned_stats}
  !IMPORTANT!

  If a customer has over a 40% risk of churning, generate a 3 sentence explanation of why they are at risk to churn.
  If a customer has less than a 40% risk of churning, generate a 3 sentence explanation of why they might not be at risk to churn.
  Your explanation should be based on the customer's information and the summary statistics for the churned and non-churned customers, and the feature importances provided.
  Don't mention the probability of churning, or the machine learning model, or say anything like 'Based on the machine learning model's prediction and top 10 most important features...', just explain the prediction.
  Don't mention or repeat the process you used to make the prediction, just explain the prediction.
  Don't use let's or other interactive language, just explain the prediction.
  Make 3 sentences have a 1 line break between them. Don't use any other line breaks or no line breaks at all.
  Convert any features names to their actual names, not the encoded ones, in plain text English.
  """
    resp = _client().chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content or ""


def generate_email(
    probability: float, input_dict: dict, explanation: str, surname: str
) -> str:
    pct = round(probability * 100, 1)
    prompt = f"""
    You are an expert email writer at a bank, where you specialize in writing emails to customers for ensuring that the customers stay with the bank and are incetivized to stay with various offers.
    You noticed that customer {surname} has a {pct}% chance of churning.
    Here is the information about the customer:
    {input_dict}
    Here is the explanation for the prediction:
    {explanation}
    Generate an email to the customer based on the above information and explanation about them to ask them to stay with the bank if they are at risk of churning, or offer them more incentives so they that remain loyal customers to the bank.
    Make sure to list out a set of incentives to stay based on their information, in bullet points format and line breaks after each bullet point, dont format anything other than what are needed to create bullet points. Don't ever mention the probability of churning, or the machine learning model, or say anything like 'Based on the machine learning model's prediction and top 10 most important features...'.
    """
    resp = _client().chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content or ""
