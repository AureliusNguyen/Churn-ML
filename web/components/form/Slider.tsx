"use client";

import { useId } from "react";

type Props = {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  format?: (v: number) => string;
  unit?: string;
};

export function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  format,
  unit,
}: Props) {
  const id = useId();
  const display = format ? format(value) : value.toString();
  const pct = ((value - min) / (max - min)) * 100;

  return (
    <div className="group">
      <div className="mb-1.5 flex items-baseline justify-between">
        <label
          htmlFor={id}
          className="text-[12px] uppercase tracking-[0.14em] text-graph"
        >
          {label}
        </label>
        <div className="font-mono text-sm tabular text-ink">
          {display}
          {unit ? <span className="ml-1 text-mute">{unit}</span> : null}
        </div>
      </div>
      <div className="relative h-[28px]">
        <div className="absolute left-0 right-0 top-1/2 h-[2px] -translate-y-1/2 bg-bone" />
        <div
          className="absolute left-0 top-1/2 h-[2px] -translate-y-1/2 bg-ink"
          style={{ width: `${pct}%` }}
        />
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="absolute inset-0 w-full cursor-pointer appearance-none bg-transparent
                     [&::-webkit-slider-thumb]:appearance-none
                     [&::-webkit-slider-thumb]:h-[18px]
                     [&::-webkit-slider-thumb]:w-[18px]
                     [&::-webkit-slider-thumb]:rounded-full
                     [&::-webkit-slider-thumb]:bg-cream
                     [&::-webkit-slider-thumb]:border-[1.5px]
                     [&::-webkit-slider-thumb]:border-ink
                     [&::-webkit-slider-thumb]:shadow-sm
                     [&::-webkit-slider-thumb]:cursor-grab
                     [&::-webkit-slider-thumb]:transition-transform
                     [&::-webkit-slider-thumb]:hover:scale-110
                     [&::-moz-range-thumb]:h-[18px]
                     [&::-moz-range-thumb]:w-[18px]
                     [&::-moz-range-thumb]:rounded-full
                     [&::-moz-range-thumb]:bg-cream
                     [&::-moz-range-thumb]:border-[1.5px]
                     [&::-moz-range-thumb]:border-ink
                     [&::-moz-range-thumb]:cursor-grab"
        />
      </div>
    </div>
  );
}
