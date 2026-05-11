"use client";

import {
  FIXTURE_CUSTOMER,
  FIXTURE_PREDICTION,
  FIXTURE_SHAP,
} from "./fixtures";
import type {
  CustomerInput,
  CustomerSearchHit,
  PredictResponse,
  ShapResponse,
} from "./types";

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api/proxy${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json() as Promise<T>;
}

async function withFallback<T>(real: () => Promise<T>, fixture: T): Promise<T> {
  try {
    return await real();
  } catch (err) {
    if (process.env.NODE_ENV !== "production") {
      console.warn("[client-api] falling back to fixture:", err);
      return fixture;
    }
    throw err;
  }
}

// Tiny what-if simulator so sliders react in fixture mode.
// Heuristic only -- the real backend always wins when reachable.
function simulatePredict(input: CustomerInput): PredictResponse {
  const fx = FIXTURE_PREDICTION;
  let p = fx.advanced_avg;
  // Active member is the strongest down-shift in the real model
  if (!input.is_active_member) p += 0.06;
  else p -= 0.04;
  if (input.num_products >= 3) p += 0.08;
  if (input.num_products === 1) p += 0.04;
  if (input.location === "Germany") p += 0.05;
  if (input.balance === 0) p += 0.03;
  if (input.age > 55) p += 0.04;
  if (input.age < 30) p -= 0.04;
  if (input.has_credit_card) p -= 0.015;
  p = Math.max(0.02, Math.min(0.98, p));
  const scale = p / fx.advanced_avg;
  return {
    basic: Object.fromEntries(
      Object.entries(fx.basic).map(([k, v]) => [
        k,
        Math.max(0.01, Math.min(0.99, v * scale)),
      ])
    ),
    advanced: Object.fromEntries(
      Object.entries(fx.advanced).map(([k, v]) => [
        k,
        Math.max(0.01, Math.min(0.99, v * scale)),
      ])
    ),
    basic_avg: Math.max(0.01, Math.min(0.99, fx.basic_avg * scale)),
    advanced_avg: p,
  };
}

function simulateShap(input: CustomerInput): ShapResponse {
  const fx = FIXTURE_SHAP;
  const adjusted = fx.shap_values.map((s) => {
    let c = s.contribution;
    if (s.feature === "Age") {
      c = (input.age - 38) * 0.025;
    } else if (s.feature === "NumOfProducts") {
      c = (input.num_products - 1.6) * 0.18;
    } else if (s.feature === "IsActiveMember") {
      c = input.is_active_member ? -0.22 : 0.08;
    } else if (s.feature === "Balance") {
      c = input.balance > 0 ? -0.1 : 0.31;
    } else if (s.feature === "Geography_Germany") {
      c = input.location === "Germany" ? 0.22 : -0.12;
    } else if (s.feature === "CreditScore") {
      c = (650 - input.credit_score) / 750;
    }
    return { ...s, value: pickValue(s.feature, input, s.value), contribution: c };
  });
  const pred = simulatePredict(input).advanced_avg;
  return {
    base_value: fx.base_value,
    expected_prob: fx.expected_prob,
    predicted_prob: pred,
    shap_values: adjusted,
  };
}

function pickValue(feat: string, input: CustomerInput, fallback: number): number {
  switch (feat) {
    case "Age":
      return input.age;
    case "NumOfProducts":
      return input.num_products;
    case "IsActiveMember":
      return input.is_active_member ? 1 : 0;
    case "Balance":
      return input.balance;
    case "CreditScore":
      return input.credit_score;
    case "Tenure":
      return input.tenure;
    case "HasCrCard":
      return input.has_credit_card ? 1 : 0;
    case "EstimatedSalary":
      return input.estimated_salary;
    case "Geography_Germany":
      return input.location === "Germany" ? 1 : 0;
    case "Geography_France":
      return input.location === "France" ? 1 : 0;
    case "Geography_Spain":
      return input.location === "Spain" ? 1 : 0;
    case "Gender_Female":
      return input.gender === "Female" ? 1 : 0;
    case "Gender_Male":
      return input.gender === "Male" ? 1 : 0;
    default:
      return fallback;
  }
}

export const clientApi = {
  searchCustomers: (q: string) =>
    withFallback<CustomerSearchHit[]>(
      () =>
        fetchJson<CustomerSearchHit[]>(
          `/customers?q=${encodeURIComponent(q)}&limit=12`
        ),
      q
        ? [
            {
              customer_id: FIXTURE_CUSTOMER.customer_id,
              surname: FIXTURE_CUSTOMER.surname,
              location: FIXTURE_CUSTOMER.location,
              age: FIXTURE_CUSTOMER.age,
            },
          ]
        : []
    ),

  predict: (input: CustomerInput) =>
    withFallback(
      () =>
        fetchJson<PredictResponse>("/predict", {
          method: "POST",
          body: JSON.stringify(input),
        }),
      simulatePredict(input)
    ),

  shap: (input: CustomerInput) =>
    withFallback(
      () =>
        fetchJson<ShapResponse>("/shap", {
          method: "POST",
          body: JSON.stringify(input),
        }),
      simulateShap(input)
    ),

  email: (
    customer: CustomerInput,
    surname: string,
    probability: number,
    explanation: string
  ) =>
    withFallback<{ text: string }>(
      () =>
        fetchJson<{ text: string }>("/email", {
          method: "POST",
          body: JSON.stringify({ customer, surname, probability, explanation }),
        }),
      {
        text: `Dear ${surname},\n\nWe noticed it's been a quiet stretch since we last connected, and we'd like to make sure you're getting the most out of your relationship with us.\n\nAs a token of appreciation, we'd like to extend:\n\n* A first-year fee waiver on the premium current account\n* A 0.50% rate uplift on your next deposit\n* A dedicated relationship manager available by direct line\n* Early access to mortgage rates and partner offers\n\nReply to this note any time, or call us to discuss.\n\nWith thanks,\nThe Retention Desk`,
      }
    ),
};
