"use client";

import { motion, useScroll, useTransform } from "motion/react";
import { useRef, type ReactNode } from "react";

/**
 * Oversized ghost numeral / mark that parallaxes against scroll. As
 * the parent section traverses the viewport, the numeral drifts ~80px
 * opposite the scroll direction, giving the page a quiet depth.
 *
 * The component renders as a positioned span you can absolutely place
 * inside a `relative` section. Pass the position via the `position`
 * prop or via Tailwind classes.
 */
export function GhostNumeral({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start end", "end start"],
  });
  const y = useTransform(scrollYProgress, [0, 1], [80, -80]);

  return (
    <motion.span
      ref={ref}
      aria-hidden
      className={`ghost-mark ${className}`}
      style={{ y }}
    >
      {children}
    </motion.span>
  );
}
