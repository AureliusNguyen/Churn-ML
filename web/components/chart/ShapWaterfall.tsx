"use client";

import { AnimatePresence, motion } from "motion/react";
import { useMemo } from "react";

import { featureLabel, fmtPct } from "@/lib/format";
import type { ShapItem } from "@/lib/types";

type Props = {
  items: ShapItem[];
  baseProb: number;
  predictedProb: number;
  topN?: number;
};

export function ShapWaterfall({ items, baseProb, predictedProb, topN = 8 }: Props) {
  const sorted = useMemo(() => {
    return [...items]
      .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
      .slice(0, topN);
  }, [items, topN]);

  const max = useMemo(
    () => Math.max(...sorted.map((s) => Math.abs(s.contribution)), 0.0001),
    [sorted]
  );

  return (
    <div className="space-y-1">
      <div className="mb-3 flex items-baseline justify-between text-[12px] uppercase tracking-[0.14em] text-mute">
        <span>Pushes toward staying</span>
        <span>Pushes toward leaving</span>
      </div>
      <ol className="space-y-1.5">
        <AnimatePresence initial={false}>
          {sorted.map((s, i) => {
            const positive = s.contribution > 0;
            const w = (Math.abs(s.contribution) / max) * 100;
            return (
              <motion.li
                key={s.feature}
                layout
                initial={{ opacity: 0, x: positive ? 4 : -4 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                transition={{
                  duration: 0.35,
                  ease: [0.2, 0, 0, 1],
                  delay: i * 0.04,
                }}
                className="grid grid-cols-[1fr_3fr_1fr_auto] items-center gap-3"
              >
                <div className="truncate text-right text-[13px] text-graph">
                  {featureLabel(s.feature)}
                </div>
                <div className="relative flex h-[12px] items-center">
                  <div className="absolute left-1/2 top-0 h-full w-px bg-rule/40" />
                  {positive ? (
                    <motion.div
                      key={`p-${s.feature}-${s.contribution}`}
                      className="absolute left-1/2 h-full bg-terra"
                      initial={{ width: 0 }}
                      animate={{ width: `${w / 2}%` }}
                      transition={{ duration: 0.45, ease: [0.2, 0, 0, 1] }}
                    />
                  ) : (
                    <motion.div
                      key={`n-${s.feature}-${s.contribution}`}
                      className="absolute right-1/2 h-full bg-ochre"
                      initial={{ width: 0 }}
                      animate={{ width: `${w / 2}%` }}
                      transition={{ duration: 0.45, ease: [0.2, 0, 0, 1] }}
                    />
                  )}
                </div>
                <div className="font-mono text-[12px] tabular text-graph">
                  {s.contribution > 0 ? "+" : ""}
                  {s.contribution.toFixed(3)}
                </div>
                <div className="font-mono text-[11px] tabular text-mute">
                  {Number.isInteger(s.value) ? s.value : s.value.toFixed(1)}
                </div>
              </motion.li>
            );
          })}
        </AnimatePresence>
      </ol>

      <div className="mt-5 grid grid-cols-2 gap-6 border-t border-rule/30 pt-4 text-[12px] uppercase tracking-[0.14em] text-mute">
        <div>
          Baseline:{" "}
          <span className="font-mono normal-case tracking-normal text-ink">
            {fmtPct(baseProb)}
          </span>
        </div>
        <div className="text-right">
          For this customer:{" "}
          <span className="font-mono normal-case tracking-normal text-ink">
            {fmtPct(predictedProb)}
          </span>
        </div>
      </div>
    </div>
  );
}
