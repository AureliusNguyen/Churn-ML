import { SearchPicker } from "@/components/ui/SearchPicker";
import { Ticker } from "@/components/ui/Ticker";
import { fmtInt, fmtMoney, fmtPct } from "@/lib/format";
import type { DatasetSummary } from "@/lib/types";

type Props = { summary: DatasetSummary | null };

function formatToday(): string {
  // e.g. "Friday, May 10, 2026"
  return new Date().toLocaleDateString("en-US", {
    weekday: "long",
    month: "long",
    day: "numeric",
    year: "numeric",
  });
}

export function Masthead({ summary }: Props) {
  const tickerItems = summary
    ? [
        { label: "Customers", value: fmtInt(summary.total) },
        { label: "Churn rate", value: fmtPct(summary.churn_rate) },
        { label: "Mean tenure", value: `${summary.mean_tenure.toFixed(1)} yr` },
        { label: "Mean balance", value: fmtMoney(summary.mean_balance) },
        { label: "Mean credit", value: summary.mean_credit_score.toFixed(0) },
        ...Object.entries(summary.geographies).map(([g, n]) => ({
          label: g,
          value: fmtInt(n),
        })),
      ]
    : [{ label: "The Churn Report", value: "Vol. I" }];

  const today = formatToday();

  return (
    <header className="pt-6">
      {/* Top hairline + edition row */}
      <div className="border-t-2 border-ink" />
      <div className="flex items-center justify-between gap-6 border-b border-rule/40 py-1.5 text-[10px] uppercase tracking-[0.32em] text-graph">
        <span className="font-mono tabular normal-case tracking-[0.18em] text-mute">
          est. 2026
        </span>
        <span className="hidden sm:inline">{today}</span>
        <span className="font-mono tabular normal-case tracking-[0.18em] text-mute">
          one cent
        </span>
      </div>

      {/* Title row */}
      <div className="flex items-end justify-between gap-6 pt-5 pb-3">
        <div className="flex-1">
          <div className="font-display text-[44px] leading-[0.95] tracking-[-0.02em] sm:text-[64px]">
            The Churn Report
          </div>
          <div className="mt-2 flex flex-wrap items-baseline gap-x-4 gap-y-1 text-[10px] uppercase tracking-[0.32em] text-graph">
            <span>Vol. I</span>
            <span aria-hidden className="text-mute">/</span>
            <span>Issue 01</span>
            <span aria-hidden className="text-mute">/</span>
            <span>Customer Risk &amp; Retention</span>
            <span aria-hidden className="text-mute">/</span>
            <span className="text-terra">Daily edition</span>
          </div>
        </div>
        <div className="hidden flex-none lg:block">
          <SearchPicker />
        </div>
      </div>

      {/* Double rule */}
      <div className="border-t-2 border-ink" />
      <div className="mt-[3px] border-t border-ink" />

      {/* Mobile search */}
      <div className="mt-4 lg:hidden">
        <SearchPicker />
      </div>

      {/* Editorial ticker */}
      <div className="mt-2">
        <Ticker items={tickerItems} />
      </div>
    </header>
  );
}
