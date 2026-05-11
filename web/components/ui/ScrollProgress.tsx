"use client";

import { motion, useScroll, useSpring } from "motion/react";

/**
 * Fixed thin terra-coloured rule down the left margin. Fills from top
 * to bottom as the page scrolls. Editorial detail (FT / NYT style).
 * Hidden on small screens to keep the margin clean.
 */
export function ScrollProgress() {
  const { scrollYProgress } = useScroll();
  const scaleY = useSpring(scrollYProgress, {
    stiffness: 120,
    damping: 26,
    mass: 0.4,
  });

  return (
    <div
      aria-hidden
      className="pointer-events-none fixed left-3 top-0 z-40 hidden h-screen w-[2px] sm:block"
      style={{ background: "rgba(196, 69, 54, 0.12)" }}
    >
      <motion.div
        className="absolute inset-x-0 top-0 bg-terra"
        style={{ height: "100%", transformOrigin: "top", scaleY }}
      />
    </div>
  );
}
