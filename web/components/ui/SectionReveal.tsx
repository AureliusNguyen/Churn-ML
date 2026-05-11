"use client";

import { motion } from "motion/react";
import type { ReactNode } from "react";

/**
 * Wraps a section in a scroll-triggered fade + rise. Fires once when
 * the section first enters the viewport. Replaces the previous one-
 * shot first-paint stagger so each section gets its moment as you
 * read down the page.
 */
export function SectionReveal({
  children,
  className,
  delay = 0,
}: {
  children: ReactNode;
  className?: string;
  delay?: number;
}) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 28 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.18 }}
      transition={{
        duration: 0.85,
        ease: [0.2, 0, 0, 1],
        delay,
      }}
    >
      {children}
    </motion.div>
  );
}
