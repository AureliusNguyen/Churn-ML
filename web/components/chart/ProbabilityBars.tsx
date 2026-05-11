"use client";

import { motion } from "motion/react";
import { fmtPct } from "@/lib/format";

type Props = {
  probabilities: Record<string, number>;
  accent?: "terra" | "ochre";
};

export function ProbabilityBars({ probabilities, accent = "terra" }: Props) {
  const entries = Object.entries(probabilities);
  const max = 1; // show full 0..100% scale so the bars are honest
  const accentClass =
    accent === "terra" ? "bg-terra" : "bg-ochre";

  return (
    <div className="space-y-3">
      {entries.map(([name, p], i) => (
        <div key={name} className="grid grid-cols-[1fr_auto] items-baseline gap-3">
          <div className="min-w-0">
            <div className="text-[13px] uppercase tracking-[0.14em] text-graph">
              {name}
            </div>
            <div className="relative mt-1.5 h-[6px] w-full bg-bone">
              <motion.div
                className={`absolute inset-y-0 left-0 ${accentClass}`}
                initial={{ width: 0 }}
                animate={{ width: `${(p / max) * 100}%` }}
                transition={{
                  duration: 0.7,
                  ease: [0.2, 0, 0, 1],
                  delay: i * 0.06,
                }}
              />
            </div>
          </div>
          <div className="font-mono text-sm tabular text-ink">{fmtPct(p, 1)}</div>
        </div>
      ))}
    </div>
  );
}
