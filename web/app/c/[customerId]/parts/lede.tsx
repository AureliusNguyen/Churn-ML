import { Gauge } from "@/components/chart/Gauge";
import { GhostNumeral } from "@/components/ui/GhostNumeral";
import { SectionReveal } from "@/components/ui/SectionReveal";

type Props = {
  surname: string;
  customerId?: number;
  probability: number;
  explanation: string;
};

function leadingSentence(text: string) {
  if (!text) return "";
  const trimmed = text.trim().split("\n").find((l) => l.trim().length > 0) || "";
  return trimmed.length > 240 ? trimmed.slice(0, 237) + "..." : trimmed;
}

export function Lede({ surname, probability, explanation }: Props) {
  const pct = (probability * 100).toFixed(1);
  const verdict =
    probability >= 0.6 ? "very likely" : probability >= 0.4 ? "likely" : "unlikely";
  const lead = leadingSentence(explanation);

  // Ghost numeral behind the gauge -- the integer percent, like a magazine
  // cover number sitting behind the real chart.
  const ghostNum = Math.round(probability * 100).toString();

  return (
    <SectionReveal>
      <section className="relative grid grid-cols-1 gap-6 pt-12 pb-10 lg:grid-cols-[1.55fr_1fr] lg:items-center lg:gap-10">
        <div className="lg:col-span-2">
          <div className="text-[10px] uppercase tracking-[0.32em] text-mute">
            The verdict <span className="text-terra">/ Section I</span>
          </div>
        </div>

        <div>
          <h1 className="font-display text-[48px] leading-[0.95] tracking-[-0.02em] sm:text-[68px] lg:text-[88px]">
            <span className="block">{surname}</span>
            <span className="block italic text-graph">
              is <span className="not-italic text-ink">{verdict}</span>{" "}
              <span className="not-italic">to leave.</span>
            </span>
          </h1>

          <div className="mt-7 max-w-[58ch] text-[17px] leading-[1.6] text-graph dropcap">
            {lead ? (
              <>
                {lead}{" "}
                <span className="text-mute">
                  The model puts the probability at{" "}
                  <span className="font-mono tabular text-ink">{pct}%</span>.
                </span>
              </>
            ) : (
              <>
                {`The model puts the probability of ${surname} leaving at ${pct}%. `}
                The full explanation will appear here as soon as the language model
                finishes drafting it.
              </>
            )}
          </div>
        </div>

        <div className="relative flex min-h-[300px] items-center justify-center lg:justify-end">
          <GhostNumeral className="right-[-12px] top-[-32px] hidden text-[260px] sm:block lg:text-[320px]">
            {ghostNum}
          </GhostNumeral>
          <div className="relative">
            <Gauge
              probability={probability}
              size={300}
              label="Estimated probability of churn"
            />
          </div>
        </div>
      </section>
    </SectionReveal>
  );
}
