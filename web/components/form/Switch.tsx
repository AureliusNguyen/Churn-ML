"use client";

type Props = {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  hint?: string;
};

export function Switch({ label, checked, onChange, hint }: Props) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="flex w-full items-center justify-between border-b border-rule/30 py-2.5 text-left transition-colors hover:bg-bone/40"
    >
      <div>
        <div className="text-sm text-ink">{label}</div>
        {hint && <div className="text-[12px] text-mute">{hint}</div>}
      </div>
      <span
        aria-hidden
        className={`relative h-[22px] w-[40px] flex-none rounded-full border border-ink transition-colors duration-200 ${
          checked ? "bg-ink" : "bg-cream"
        }`}
      >
        <span
          className={`absolute top-1/2 h-[14px] w-[14px] -translate-y-1/2 rounded-full transition-all duration-200 ease-out ${
            checked ? "left-[22px] bg-cream" : "left-[3px] bg-ink"
          }`}
        />
      </span>
    </button>
  );
}
