type Props = {
  label: string;
  value: string;
  meta?: string;
};

export function StatCard({ label, value, meta }: Props) {
  return (
    <div className="border-t border-rule pt-3">
      <div className="text-[11px] uppercase tracking-[0.18em] text-mute">{label}</div>
      <div className="mt-1 font-mono text-3xl tabular tracking-tight text-ink">
        {value}
      </div>
      {meta && (
        <div className="mt-1 text-[12px] italic text-graph">{meta}</div>
      )}
    </div>
  );
}
