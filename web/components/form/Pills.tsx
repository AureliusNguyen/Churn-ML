"use client";

type Props<T extends string> = {
  label: string;
  value: T;
  options: readonly T[];
  onChange: (v: T) => void;
};

export function Pills<T extends string>({ label, value, options, onChange }: Props<T>) {
  return (
    <div>
      <div className="mb-1.5 text-[12px] uppercase tracking-[0.14em] text-graph">
        {label}
      </div>
      <div role="radiogroup" aria-label={label} className="flex border border-ink">
        {options.map((opt, i) => {
          const active = opt === value;
          return (
            <button
              key={opt}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => onChange(opt)}
              className={`flex-1 px-3 py-2 text-sm transition-colors
                ${i > 0 ? "border-l border-ink" : ""}
                ${active ? "bg-ink text-cream" : "bg-transparent text-ink hover:bg-bone"}`}
            >
              {opt}
            </button>
          );
        })}
      </div>
    </div>
  );
}
