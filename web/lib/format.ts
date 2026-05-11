export const fmtPct = (p: number, digits = 1) =>
  `${(p * 100).toFixed(digits)}%`;

export const fmtMoney = (n: number) =>
  n === 0
    ? "$0"
    : new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 0,
      }).format(n);

export const fmtInt = (n: number) =>
  new Intl.NumberFormat("en-US").format(Math.round(n));

export const fmtPp = (delta: number, digits = 1) => {
  const sign = delta > 0 ? "+" : "";
  return `${sign}${(delta * 100).toFixed(digits)} pp`;
};

const FEATURE_LABELS: Record<string, string> = {
  CreditScore: "Credit score",
  Age: "Age",
  Tenure: "Tenure",
  Balance: "Balance",
  NumOfProducts: "Products held",
  HasCrCard: "Has credit card",
  IsActiveMember: "Active member",
  EstimatedSalary: "Estimated salary",
  Geography_France: "Geography: France",
  Geography_Germany: "Geography: Germany",
  Geography_Spain: "Geography: Spain",
  Gender_Female: "Gender: Female",
  Gender_Male: "Gender: Male",
  CLV: "Customer lifetime value",
  TenureAgeRatio: "Tenure / Age ratio",
  AgeGroup_MiddleAged: "Age group: middle-aged",
  AgeGroup_Senior: "Age group: senior",
  AgeGroup_Elderly: "Age group: elderly",
};

export const featureLabel = (k: string) => FEATURE_LABELS[k] ?? k;
