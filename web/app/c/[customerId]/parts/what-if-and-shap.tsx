"use client";

import { motion, useReducedMotion, useSpring, useTransform } from "motion/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Gauge } from "@/components/chart/Gauge";
import { ShapWaterfall } from "@/components/chart/ShapWaterfall";
import { Pills } from "@/components/form/Pills";
import { Slider } from "@/components/form/Slider";
import { Switch } from "@/components/form/Switch";
import { GhostNumeral } from "@/components/ui/GhostNumeral";
import { SectionReveal } from "@/components/ui/SectionReveal";
import { clientApi } from "@/lib/client-api";
import { featureLabel, fmtMoney, fmtPct, fmtPp } from "@/lib/format";
import type { CustomerInput, PredictResponse, ShapResponse } from "@/lib/types";

type Props = {
  customerId: number;
  surname: string;
  baselineInput: CustomerInput;
  baselinePrediction: PredictResponse;
};

const GEOGRAPHIES = ["France", "Germany", "Spain"] as const;
const GENDERS = ["Male", "Female"] as const;

export function WhatIfAndShap({
  surname,
  baselineInput,
  baselinePrediction,
}: Props) {
  const [input, setInput] = useState<CustomerInput>(baselineInput);
  const [prediction, setPrediction] = useState<PredictResponse>(baselinePrediction);
  const [shap, setShap] = useState<ShapResponse | null>(null);
  const [pending, setPending] = useState(false);
  const reqId = useRef(0);
  const reduced = useReducedMotion();

  // Animated probability number
  const probSpring = useSpring(reduced ? prediction.advanced_avg : 0, {
    stiffness: 90,
    damping: 22,
    mass: 0.7,
  });
  useEffect(() => {
    probSpring.set(prediction.advanced_avg);
  }, [probSpring, prediction.advanced_avg]);
  const display = useTransform(probSpring, (v) => `${(v * 100).toFixed(1)}%`);

  // Debounced fetch
  useEffect(() => {
    const t = setTimeout(async () => {
      const id = ++reqId.current;
      setPending(true);
      try {
        const [p, s] = await Promise.all([
          clientApi.predict(input),
          clientApi.shap(input),
        ]);
        if (id === reqId.current) {
          setPrediction(p);
          setShap(s);
        }
      } catch {
        // network / api blip; keep prior values
      } finally {
        if (id === reqId.current) setPending(false);
      }
    }, 200);
    return () => clearTimeout(t);
  }, [input]);

  // Initial SHAP fetch on mount (server only sent prediction).
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const s = await clientApi.shap(baselineInput);
        if (!cancelled) setShap(s);
      } catch {
        // ignore
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const reset = useCallback(() => setInput(baselineInput), [baselineInput]);
  const update = <K extends keyof CustomerInput>(k: K, v: CustomerInput[K]) =>
    setInput((prev) => ({ ...prev, [k]: v }));

  const delta = prediction.advanced_avg - baselinePrediction.advanced_avg;
  const deltaPp = delta * 100;
  const deltaPositive = deltaPp > 0.05;
  const deltaNegative = deltaPp < -0.05;

  const topDriver = useMemo(() => {
    if (!shap) return null;
    return [...shap.shap_values].sort(
      (a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)
    )[0];
  }, [shap]);

  return (
    <SectionReveal>
    <section className="relative py-12">
      <div className="mb-3 border-t border-ink/80" />
      <div className="mb-3 border-t border-ink/80" />
      <GhostNumeral className="right-0 top-2 text-[180px] italic sm:text-[220px]">
        IV
      </GhostNumeral>

      <div className="text-[10px] uppercase tracking-[0.32em] text-mute">
        The what-if desk <span className="text-terra">/ Section IV</span>
      </div>
      <div className="grid grid-cols-1 gap-x-12 gap-y-10 lg:grid-cols-[1.1fr_1fr] lg:items-start">
        <div>
          <h2 className="font-display mt-2 max-w-[20ch] text-[32px] tracking-[-0.01em] sm:text-[40px]">
            What if {surname} were different?
          </h2>
          <p className="mt-3 max-w-[58ch] text-[15px] leading-[1.6] text-graph">
            Move any slider, flip any switch. The probability and the
            attribution chart update live. Use it to ask, &ldquo;what would it take
            to bring this customer back into the green?&rdquo;
          </p>

          <div className="mt-8 grid gap-6 sm:grid-cols-2">
            <Pills
              label="Geography"
              value={input.location}
              options={GEOGRAPHIES}
              onChange={(v) => update("location", v)}
            />
            <Pills
              label="Gender"
              value={input.gender}
              options={GENDERS}
              onChange={(v) => update("gender", v)}
            />
            <Slider
              label="Age"
              value={input.age}
              onChange={(v) => update("age", v)}
              min={18}
              max={92}
              unit="yr"
            />
            <Slider
              label="Tenure"
              value={input.tenure}
              onChange={(v) => update("tenure", v)}
              min={0}
              max={10}
              unit="yr"
            />
            <Slider
              label="Credit score"
              value={input.credit_score}
              onChange={(v) => update("credit_score", v)}
              min={300}
              max={850}
            />
            <Slider
              label="Number of products"
              value={input.num_products}
              onChange={(v) => update("num_products", v)}
              min={1}
              max={4}
            />
            <Slider
              label="Balance"
              value={input.balance}
              onChange={(v) => update("balance", v)}
              min={0}
              max={250000}
              step={500}
              format={fmtMoney}
            />
            <Slider
              label="Estimated salary"
              value={input.estimated_salary}
              onChange={(v) => update("estimated_salary", v)}
              min={0}
              max={200000}
              step={500}
              format={fmtMoney}
            />
          </div>

          <div className="mt-6">
            <Switch
              label="Active member"
              checked={input.is_active_member}
              onChange={(v) => update("is_active_member", v)}
              hint="Engaged with the bank in the last 12 months"
            />
            <Switch
              label="Has credit card"
              checked={input.has_credit_card}
              onChange={(v) => update("has_credit_card", v)}
            />
          </div>

          <div className="mt-6 flex items-center gap-3 text-[13px]">
            <button
              type="button"
              onClick={reset}
              className="text-terra underline-offset-2 hover:underline"
            >
              Restore {surname}&rsquo;s actual values
            </button>
            <span className="text-mute">&middot;</span>
            <span className={`font-mono tabular ${pending ? "text-mute" : "text-graph"}`}>
              {pending ? "computing..." : "synced"}
            </span>
          </div>
        </div>

        {/* Right column: live probability + delta + gauge + SHAP */}
        <div className="lg:sticky lg:top-6 lg:self-start">
          <div className="relative border-2 border-ink bg-paper/40 p-6">
            {/* Corner registration marks, printer style */}
            <span aria-hidden className="absolute left-[-1px] top-[-1px] block h-[10px] w-[10px] border-l-2 border-t-2 border-ink" />
            <span aria-hidden className="absolute right-[-1px] top-[-1px] block h-[10px] w-[10px] border-r-2 border-t-2 border-ink" />
            <span aria-hidden className="absolute left-[-1px] bottom-[-1px] block h-[10px] w-[10px] border-l-2 border-b-2 border-ink" />
            <span aria-hidden className="absolute right-[-1px] bottom-[-1px] block h-[10px] w-[10px] border-r-2 border-b-2 border-ink" />

            <div className="flex items-baseline justify-between">
              <div className="text-[10px] uppercase tracking-[0.28em] text-mute">
                Probability of churn
              </div>
              <div
                className={`text-[11px] uppercase tracking-[0.18em] ${
                  deltaPositive ? "text-terra" : deltaNegative ? "text-ochre" : "text-mute"
                }`}
              >
                {Math.abs(deltaPp) < 0.05
                  ? "= baseline"
                  : `${fmtPp(delta)} from actual`}
              </div>
            </div>

            <motion.div
              className="mt-3 flex items-baseline gap-1"
              animate={{ opacity: pending ? 0.55 : 1 }}
              transition={{ duration: 0.18 }}
            >
              <motion.span className="font-display text-[88px] leading-[0.9] tracking-[-0.03em] tabular-nums">
                {display}
              </motion.span>
            </motion.div>

            <div className="mt-4 flex justify-center">
              <Gauge probability={prediction.advanced_avg} size={240} />
            </div>
          </div>

          <div className="mt-8">
            <div className="mb-3 flex items-baseline justify-between">
              <h3 className="font-display text-xl">Why the model says so</h3>
              <span className="text-[11px] uppercase tracking-[0.18em] text-mute">
                Top contributors
              </span>
            </div>
            {shap ? (
              <>
                <ShapWaterfall
                  items={shap.shap_values}
                  baseProb={shap.expected_prob}
                  predictedProb={shap.predicted_prob}
                />
                {topDriver && (
                  <p className="mt-5 max-w-[52ch] font-display text-[15px] italic leading-[1.55] text-graph">
                    The model leans on{" "}
                    <span className="not-italic text-ink">
                      {featureLabel(topDriver.feature).toLowerCase()}
                    </span>{" "}
                    more than anything else for this customer
                    {topDriver.contribution > 0
                      ? ", and that pushes the prediction toward leaving."
                      : ", and that pulls the prediction toward staying."}
                  </p>
                )}
              </>
            ) : (
              <div className="font-mono text-sm text-mute">
                computing attribution...
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
    </SectionReveal>
  );
}
