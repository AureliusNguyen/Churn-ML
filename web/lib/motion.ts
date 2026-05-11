import type { Easing, Transition } from "motion/react";

export const easeEditorial: Easing = [0.2, 0, 0, 1];

export const tFast: Transition = { duration: 0.2, ease: easeEditorial };
export const tNormal: Transition = { duration: 0.4, ease: easeEditorial };
export const tSlow: Transition = { duration: 0.7, ease: easeEditorial };

export const fadeUp = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  transition: tNormal,
};

export const fadeIn = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  transition: tNormal,
};

export const stagger = (delayPerChild = 0.06): Transition => ({
  staggerChildren: delayPerChild,
});
