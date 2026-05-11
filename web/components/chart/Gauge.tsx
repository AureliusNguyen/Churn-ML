"use client";

import {
  motion,
  useMotionValueEvent,
  useReducedMotion,
  useSpring,
  useTransform,
} from "motion/react";
import { useEffect, useRef } from "react";

type Props = {
  /** 0..1 */
  probability: number;
  size?: number;
  label?: string;
};

const ARC_RADIUS = 88;
const ARC_STROKE = 14;
const SWEEP = 220; // degrees, from -110 to +110

function polar(angleDeg: number, radius: number) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return { x: radius * Math.cos(rad), y: radius * Math.sin(rad) };
}

function arcPath(startDeg: number, endDeg: number, radius: number) {
  const start = polar(startDeg, radius);
  const end = polar(endDeg, radius);
  const large = endDeg - startDeg > 180 ? 1 : 0;
  return `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${large} 1 ${end.x} ${end.y}`;
}

export function Gauge({ probability, size = 240, label }: Props) {
  const reduced = useReducedMotion();
  const target = Math.min(Math.max(probability, 0), 1);
  const spring = useSpring(reduced ? target : 0, {
    stiffness: 80,
    damping: 22,
    mass: 0.7,
  });

  useEffect(() => {
    spring.set(target);
  }, [spring, target]);

  // map 0..1 -> -110..+110 deg
  const angleStart = -SWEEP / 2;
  const angleEnd = SWEEP / 2;
  const display = useTransform(spring, (v) => `${(v * 100).toFixed(1)}%`);

  // Drive the SVG transform attribute imperatively on every spring frame.
  // Motion does not reliably subscribe MotionValue<string> to SVG attribute
  // props, and CSS transform-origin behaves unpredictably on <g> across
  // browsers, so we write the attribute directly via a ref. The fallback
  // value below is the rotation at v=0 (start of sweep) so SSR markup is
  // valid before the first client tick.
  const needleRef = useRef<SVGGElement>(null);
  useMotionValueEvent(spring, "change", (v) => {
    const angle = angleStart + v * SWEEP;
    needleRef.current?.setAttribute("transform", `rotate(${angle})`);
  });

  const initialAngle = angleStart + (reduced ? target : 0) * SWEEP;

  // arc segments: low (0-30), mid (30-60), high (60-100)
  const lowEnd = angleStart + 0.3 * SWEEP;
  const midEnd = angleStart + 0.6 * SWEEP;

  const w = size;
  const h = size * 0.62;

  return (
    <div className="relative inline-flex flex-col items-center" style={{ width: w }}>
      <svg
        width={w}
        height={h}
        viewBox={`${-w / 2} ${-h + 20} ${w} ${h}`}
        aria-label={label || "Churn probability gauge"}
        role="img"
      >
        {/* Background track */}
        <path
          d={arcPath(angleStart, angleEnd, ARC_RADIUS)}
          fill="none"
          stroke="var(--color-bone)"
          strokeWidth={ARC_STROKE}
          strokeLinecap="round"
        />
        {/* Low band */}
        <path
          d={arcPath(angleStart, lowEnd, ARC_RADIUS)}
          fill="none"
          stroke="var(--color-ochre)"
          strokeOpacity={0.55}
          strokeWidth={ARC_STROKE}
          strokeLinecap="round"
        />
        {/* Mid band */}
        <path
          d={arcPath(lowEnd, midEnd, ARC_RADIUS)}
          fill="none"
          stroke="var(--color-ochre)"
          strokeWidth={ARC_STROKE}
        />
        {/* High band */}
        <path
          d={arcPath(midEnd, angleEnd, ARC_RADIUS)}
          fill="none"
          stroke="var(--color-terra)"
          strokeWidth={ARC_STROKE}
          strokeLinecap="round"
        />

        {/* Tick marks at 0, 30, 60, 100 */}
        {[0, 0.3, 0.6, 1].map((t) => {
          const a = angleStart + t * SWEEP;
          const inner = polar(a, ARC_RADIUS - ARC_STROKE / 2 - 4);
          const outer = polar(a, ARC_RADIUS + ARC_STROKE / 2 + 6);
          return (
            <line
              key={t}
              x1={inner.x}
              y1={inner.y}
              x2={outer.x}
              y2={outer.y}
              stroke="var(--color-ink)"
              strokeWidth={1}
            />
          );
        })}

        {/* Needle: ref-driven SVG transform (see useMotionValueEvent above) */}
        <g ref={needleRef} transform={`rotate(${initialAngle})`}>
          <line
            x1={0}
            y1={0}
            x2={0}
            y2={-(ARC_RADIUS - 6)}
            stroke="var(--color-ink)"
            strokeWidth={2.5}
            strokeLinecap="round"
          />
          <circle cx={0} cy={0} r={6} fill="var(--color-ink)" />
        </g>
      </svg>
      <div className="mt-1 text-center">
        <motion.div className="font-mono text-3xl tabular tracking-tight">
          {display}
        </motion.div>
        {label && (
          <div className="mt-1 text-[11px] uppercase tracking-[0.18em] text-mute">
            {label}
          </div>
        )}
      </div>
    </div>
  );
}
