import { ProbabilityBars } from "@/components/chart/ProbabilityBars";
import { Asterism } from "@/components/ui/Asterism";
import { GhostNumeral } from "@/components/ui/GhostNumeral";
import { SectionReveal } from "@/components/ui/SectionReveal";
import type { PredictResponse } from "@/lib/types";

type Props = { prediction: PredictResponse };

export function ModelView({ prediction }: Props) {
  return (
    <SectionReveal>
      <Asterism />
      <section className="relative py-10">
        <GhostNumeral className="left-[-12px] top-[-30px] text-[180px] italic sm:text-[220px]">
          III
        </GhostNumeral>

        <div className="text-[10px] uppercase tracking-[0.32em] text-mute">
          How the models see this customer <span className="text-terra">/ Section III</span>
        </div>
        <h2 className="font-display mt-2 max-w-[26ch] text-[32px] tracking-[-0.01em] sm:text-[40px]">
          Eight classifiers, two committees.
        </h2>
        <p className="mt-3 max-w-[60ch] text-[15px] leading-[1.6] text-graph">
          Each model gets the same inputs but reaches its own probability. The
          spread tells you how confident the consensus really is.
        </p>

        <div className="mt-10 grid grid-cols-1 gap-x-12 gap-y-12 lg:grid-cols-[1fr_auto_1fr]">
          <div>
            <div className="border-b border-rule pb-2">
              <div className="text-[10px] uppercase tracking-[0.22em] text-mute">
                Committee A
              </div>
              <div className="font-display mt-1 text-xl">
                Four classical learners
              </div>
            </div>
            <div className="mt-5">
              <ProbabilityBars probabilities={prediction.basic} accent="ochre" />
            </div>
            <p className="marginalia mt-5 max-w-[44ch]">
              XGBoost typically anchors the high end of this committee; the
              simpler classifiers pull the average back toward the centre.
            </p>
          </div>

          {/* Vertical column rule, like newspaper inter-column gutters */}
          <div aria-hidden className="hidden lg:block lg:w-px lg:bg-rule/40" />

          <div>
            <div className="border-b border-rule pb-2">
              <div className="text-[10px] uppercase tracking-[0.22em] text-mute">
                Committee B
              </div>
              <div className="font-display mt-1 text-xl">
                Four XGBoost variants
              </div>
            </div>
            <div className="mt-5">
              <ProbabilityBars probabilities={prediction.advanced} accent="terra" />
            </div>
            <p className="marginalia mt-5 max-w-[44ch]">
              Feature engineering and class-rebalancing tend to sharpen these
              estimates, for better or for worse on any given customer.
            </p>
          </div>
        </div>
      </section>
    </SectionReveal>
  );
}
