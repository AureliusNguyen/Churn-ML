export type CustomerInput = {
  credit_score: number;
  location: "France" | "Germany" | "Spain";
  gender: "Male" | "Female";
  age: number;
  tenure: number;
  balance: number;
  num_products: number;
  has_credit_card: boolean;
  is_active_member: boolean;
  estimated_salary: number;
};

export type CustomerDetail = CustomerInput & {
  customer_id: number;
  surname: string;
  exited: boolean;
};

export type CustomerSearchHit = {
  customer_id: number;
  surname: string;
  location: string;
  age: number;
};

export type PredictResponse = {
  basic: Record<string, number>;
  advanced: Record<string, number>;
  basic_avg: number;
  advanced_avg: number;
};

export type ShapItem = {
  feature: string;
  value: number;
  contribution: number;
};

export type ShapResponse = {
  base_value: number;
  expected_prob: number;
  predicted_prob: number;
  shap_values: ShapItem[];
};

export type DatasetSummary = {
  total: number;
  churn_rate: number;
  mean_tenure: number;
  mean_balance: number;
  mean_credit_score: number;
  geographies: Record<string, number>;
};
