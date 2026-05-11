type Item = string | { label: string; value: string };

type Props = {
  items: Item[];
};

export function Ticker({ items }: Props) {
  const doubled = [...items, ...items];
  return (
    <div className="ticker-paused relative overflow-hidden border-y border-rule/40 py-2">
      <div className="ticker flex w-max items-center gap-12 whitespace-nowrap">
        {doubled.map((item, i) => (
          <div
            key={i}
            className="flex items-baseline gap-2 text-[11px] uppercase tracking-[0.22em] text-graph"
          >
            {typeof item === "string" ? (
              <span>{item}</span>
            ) : (
              <>
                <span className="text-mute">{item.label}</span>
                <span className="font-mono text-ink tabular">{item.value}</span>
              </>
            )}
            <span aria-hidden className="text-mute">/</span>
          </div>
        ))}
      </div>
    </div>
  );
}
