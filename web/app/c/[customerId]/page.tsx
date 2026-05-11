import { notFound } from "next/navigation";

import { api } from "@/lib/api";
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

  let customer;
  try {
    customer = await api.getCustomer(id);
  } catch {
    notFound();
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

  // Server-side baseline: prediction + explanation so first paint has real numbers.
  const [prediction, summary] = await Promise.all([
    api.predict(input),
    api.summary().catch(() => null),
  ]);

  let explanation = "";
  try {
    const r = await api.explain(input, customer.surname, prediction.advanced_avg);
    explanation = r.text;
  } catch {
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
