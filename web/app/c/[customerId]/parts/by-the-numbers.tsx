import { Asterism } from "@/components/ui/Asterism";
import { GhostNumeral } from "@/components/ui/GhostNumeral";
import { SectionReveal } from "@/components/ui/SectionReveal";
import { StatCard } from "@/components/ui/StatCard";
import { fmtMoney } from "@/lib/format";
import type { CustomerDetail, PredictResponse } from "@/lib/types";

type Props = {
  customer: CustomerDetail;
  prediction: PredictResponse;
};

export function ByTheNumbers({ customer, prediction }: Props) {
  const cards = [
    {
      label: "Age",
      value: customer.age.toString(),
      meta: `${customer.gender}, in ${customer.location}`,
    },
    {
      label: "Tenure",
      value: `${customer.tenure} yr`,
      meta: customer.tenure < 3 ? "newer relationship" : "long-tenured",
    },
    {
      label: "Balance",
      value: fmtMoney(customer.balance),
      meta: customer.balance === 0 ? "no funds on deposit" : "active deposit",
    },
    {
      label: "Products",
      value: customer.num_products.toString(),
      meta: customer.is_active_member ? "active member" : "dormant member",
    },
  ];

  const basicAvgPct = (prediction.basic_avg * 100).toFixed(1);
  const advancedAvgPct = (prediction.advanced_avg * 100).toFixed(1);

  return (
    <SectionReveal>
      <Asterism />
      <section className="relative grid grid-cols-1 gap-x-10 gap-y-10 py-10 lg:grid-cols-[1.9fr_1fr]">
        <GhostNumeral className="right-[-8px] top-[-30px] text-[180px] italic sm:text-[220px]">
          II
        </GhostNumeral>

        <div>
          <div className="text-[10px] uppercase tracking-[0.32em] text-mute">
            By the numbers <span className="text-terra">/ Section II</span>
          </div>
          <h2 className="font-display mt-2 text-[28px] tracking-[-0.01em] sm:text-[34px]">
            A short profile of {customer.surname}.
          </h2>
          <div className="mt-7 grid grid-cols-2 gap-x-8 gap-y-7 sm:grid-cols-4">
            {cards.map((c) => (
              <StatCard key={c.label} {...c} />
            ))}
          </div>
        </div>

        <aside className="relative lg:border-l lg:border-rule lg:pl-8">
          <div className="text-[10px] uppercase tracking-[0.32em] text-mute">
            Two ensembles, one verdict
          </div>
          <div className="mt-3 space-y-2 font-mono text-sm tabular text-graph">
            <div className="flex items-baseline justify-between">
              <span>Basic ensemble</span>
              <span className="text-ink">{basicAvgPct}%</span>
            </div>
            <div className="flex items-baseline justify-between">
              <span>Advanced ensemble</span>
              <span className="text-ink">{advancedAvgPct}%</span>
            </div>
          </div>
          <p className="marginalia mt-4 max-w-[44ch]">
            The advanced ensemble adds three engineered features -- lifetime
            value, tenure-to-age, and an age band -- to the same input. We
            treat its average as the headline figure for this report.
          </p>
        </aside>
      </section>
    </SectionReveal>
  );
}
