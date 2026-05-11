import { notFound } from "next/navigation";

import { api } from "@/lib/api";
import { FIXTURE_CUSTOMER, FIXTURE_PREDICTION, FIXTURE_SUMMARY } from "@/lib/fixtures";
import type { CustomerInput } from "@/lib/types";

import { ByTheNumbers } from "./parts/by-the-numbers";
import { Colophon } from "./parts/colophon";
import { Lede } from "./parts/lede";
import { Masthead } from "./parts/masthead";
import { ModelView } from "./parts/model-view";
import { Pitch } from "./parts/pitch";
import { WhatIfAndShap } from "./parts/what-if-and-shap";

export const dynamic = "force-dynamic";

export default async function CustomerPage({
  params,
}: {
  params: Promise<{ customerId: string }>;
}) {
  const { customerId } = await params;
  const id = Number(customerId);
  if (!Number.isFinite(id)) notFound();

  // Defensive fetches: each call has its own try/catch so a transient
  // upstream failure on /predict or /explain doesn't 500 the whole page.
  // /customers is the only hard requirement -- a missing customer truly is
  // a 404. Everything else falls back to a fixture-equivalent.
  let customer;
  try {
    customer = await api.getCustomer(id);
  } catch (err) {
    console.error("[page] getCustomer failed:", err);
    // If the id is the well-known default, fall back to fixture so the
    // page still renders. Anything else 404s.
    if (id === FIXTURE_CUSTOMER.customer_id) {
      customer = { ...FIXTURE_CUSTOMER, customer_id: id };
    } else {
      notFound();
    }
  }

  const input: CustomerInput = {
    credit_score: customer.credit_score,
    location: customer.location as CustomerInput["location"],
    gender: customer.gender as CustomerInput["gender"],
    age: customer.age,
    tenure: customer.tenure,
    balance: customer.balance,
    num_products: customer.num_products,
    has_credit_card: customer.has_credit_card,
    is_active_member: customer.is_active_member,
    estimated_salary: customer.estimated_salary,
  };

  const [prediction, summary] = await Promise.all([
    api.predict(input).catch((err) => {
      console.error("[page] predict failed:", err);
      return FIXTURE_PREDICTION;
    }),
    api.summary().catch((err) => {
      console.error("[page] summary failed:", err);
      return FIXTURE_SUMMARY;
    }),
  ]);

  let explanation = "";
  try {
    const r = await api.explain(input, customer.surname, prediction.advanced_avg);
    explanation = r.text;
  } catch (err) {
    console.error("[page] explain failed:", err);
    explanation = "";
  }

  return (
    <main className="mx-auto max-w-[1180px] px-5 pb-24 sm:px-8">
      <Masthead summary={summary} />
      <Lede
        surname={customer.surname}
        probability={prediction.advanced_avg}
        explanation={explanation}
      />
      <ByTheNumbers customer={customer} prediction={prediction} />
      <ModelView prediction={prediction} />
      <WhatIfAndShap
        customerId={customer.customer_id}
        surname={customer.surname}
        baselineInput={input}
        baselinePrediction={prediction}
      />
      <Pitch
        customer={input}
        surname={customer.surname}
        probability={prediction.advanced_avg}
        explanation={explanation}
      />
      <Colophon />
    </main>
  );
}
