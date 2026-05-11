import {
  FIXTURE_CUSTOMER,
  FIXTURE_EXPLANATION,
  FIXTURE_PREDICTION,
  FIXTURE_SHAP,
  FIXTURE_SUMMARY,
} from "./fixtures";
import type {
  CustomerDetail,
  CustomerInput,
  CustomerSearchHit,
  DatasetSummary,
  PredictResponse,
  ShapResponse,
} from "./types";

const FASTAPI_URL =
  process.env.FASTAPI_URL || process.env.NEXT_PUBLIC_FASTAPI_URL;

/** True when no backend URL is configured -- frontend runs on fixtures. */
const FIXTURE_MODE = !FASTAPI_URL;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (!FASTAPI_URL) throw new Error("FASTAPI_URL not set");
  const res = await fetch(`${FASTAPI_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${path} failed (${res.status}): ${body}`);
  }
  return res.json() as Promise<T>;
}

/** Wraps a request with a fixture fallback for offline-frontend dev. */
async function withFallback<T>(real: () => Promise<T>, fixture: T): Promise<T> {
  if (FIXTURE_MODE) return fixture;
  try {
    return await real();
  } catch (err) {
    if (process.env.NODE_ENV !== "production") {
      console.warn("[api] falling back to fixture:", err);
      return fixture;
    }
    throw err;
  }
}

export const api = {
  health: () =>
    withFallback(() => request<{ status: string; models_loaded: number }>("/health"), {
      status: "fixture",
      models_loaded: 0,
    }),

  searchCustomers: (q: string, limit = 12) =>
    withFallback(
      () =>
        request<CustomerSearchHit[]>(
          `/customers?q=${encodeURIComponent(q)}&limit=${limit}`
        ),
      [
        {
          customer_id: FIXTURE_CUSTOMER.customer_id,
          surname: FIXTURE_CUSTOMER.surname,
          geography: FIXTURE_CUSTOMER.location,
          age: FIXTURE_CUSTOMER.age,
        },
      ]
    ),

  getCustomer: (id: number) =>
    withFallback<CustomerDetail>(
      () => request<CustomerDetail>(`/customers/${id}`),
      { ...FIXTURE_CUSTOMER, customer_id: id }
    ),

  predict: (input: CustomerInput) =>
    withFallback(
      () =>
        request<PredictResponse>("/predict", {
          method: "POST",
          body: JSON.stringify(input),
        }),
      FIXTURE_PREDICTION
    ),

  shap: (input: CustomerInput) =>
    withFallback(
      () =>
        request<ShapResponse>("/shap", {
          method: "POST",
          body: JSON.stringify(input),
        }),
      FIXTURE_SHAP
    ),

  explain: (customer: CustomerInput, surname: string, probability: number) =>
    withFallback<{ text: string }>(
      () =>
        request<{ text: string }>("/explain", {
          method: "POST",
          body: JSON.stringify({ customer, surname, probability }),
        }),
      { text: FIXTURE_EXPLANATION }
    ),

  email: (
    customer: CustomerInput,
    surname: string,
    probability: number,
    explanation: string
  ) =>
    withFallback<{ text: string }>(
      () =>
        request<{ text: string }>("/email", {
          method: "POST",
          body: JSON.stringify({ customer, surname, probability, explanation }),
        }),
      {
        text: `Dear ${surname},\n\nWe noticed it has been a while since you've made the most of your relationship with us, and we wanted to reach out personally to remind you of what's available to you.\n\nAs a token of appreciation for your continued business, we'd like to offer:\n\n* A complimentary first-year fee waiver on our premium current account\n* A 0.50% rate uplift on your next 12-month deposit\n* A dedicated relationship manager available by direct line\n* Priority access to early-bird mortgage rates and partner offers\n\nWe value your relationship and would love the chance to keep building on it. Reply to this note any time, or call us at the number below to discuss.\n\nWith thanks,\nThe Retention Desk`,
      }
    ),

  summary: () =>
    withFallback<DatasetSummary>(() => request<DatasetSummary>("/dataset/summary"), FIXTURE_SUMMARY),
};
