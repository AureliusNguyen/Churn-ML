/**
 * Mock data used when the FastAPI service is unreachable -- lets the
 * frontend render and be iterated on without standing up the Python
 * backend. Real values come from the live /api once deployed.
 *
 * Picking realistic shape numbers so the editorial copy still reads
 * naturally (high-risk customer; SHAP shows the usual suspects).
 */

import type {
  CustomerDetail,
  DatasetSummary,
  PredictResponse,
  ShapResponse,
} from "./types";

export const FIXTURE_CUSTOMER: CustomerDetail = {
  customer_id: 15634602,
  surname: "Hargrave",
  credit_score: 619,
  location: "France",
  gender: "Female",
  age: 42,
  tenure: 2,
  balance: 0,
  num_products: 1,
  has_credit_card: true,
  is_active_member: true,
  estimated_salary: 101348.88,
  exited: true,
};

export const FIXTURE_PREDICTION: PredictResponse = {
  basic: {
    XGBoost: 0.732,
    "Random Forest": 0.564,
    "K-Nearest Neighbor": 0.4,
    "Support Vector Machine": 0.61,
  },
  advanced: {
    "XGBoost with Feature Engineering": 0.748,
    "XGBoost with SMOTE": 0.812,
    "Best XGB": 0.79,
    Stacking: 0.722,
  },
  basic_avg: 0.5765,
  advanced_avg: 0.768,
};

export const FIXTURE_SHAP: ShapResponse = {
  base_value: -1.05,
  expected_prob: 0.26,
  predicted_prob: 0.79,
  shap_values: [
    { feature: "Age", value: 42, contribution: 0.62 },
    { feature: "NumOfProducts", value: 1, contribution: 0.41 },
    { feature: "IsActiveMember", value: 1, contribution: -0.18 },
    { feature: "Geography_Germany", value: 0, contribution: -0.12 },
    { feature: "Balance", value: 0, contribution: 0.31 },
    { feature: "CreditScore", value: 619, contribution: 0.09 },
    { feature: "Gender_Female", value: 1, contribution: 0.07 },
    { feature: "EstimatedSalary", value: 101348.88, contribution: -0.04 },
    { feature: "Tenure", value: 2, contribution: 0.22 },
    { feature: "CLV", value: 0, contribution: 0.05 },
    { feature: "TenureAgeRatio", value: 0.0476, contribution: 0.18 },
    { feature: "AgeGroup_MiddleAged", value: 1, contribution: -0.03 },
    { feature: "HasCrCard", value: 1, contribution: -0.02 },
  ],
};

export const FIXTURE_SUMMARY: DatasetSummary = {
  total: 10000,
  churn_rate: 0.2037,
  mean_tenure: 5.0128,
  mean_balance: 76485.89,
  mean_credit_score: 650.5288,
  geographies: { France: 5014, Germany: 2509, Spain: 2477 },
};

export const FIXTURE_EXPLANATION =
  "Hargrave shows the textbook fingerprints of a customer about to walk: she holds only one product, has a zero balance, and the relationship with the bank is only two years old. \n\nWomen in her age band churn at a noticeably higher rate than the rest of the book, and the lack of additional products means the bank has very little to offer her in terms of switching cost. \n\nWith no credit card on file and an inactive engagement pattern, there is almost no friction stopping her from moving her primary banking relationship elsewhere.";
