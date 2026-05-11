"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { clientApi } from "@/lib/client-api";
import type { CustomerSearchHit } from "@/lib/types";

export function SearchPicker() {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [hits, setHits] = useState<CustomerSearchHit[]>([]);
  const [active, setActive] = useState(0);
  const router = useRouter();
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!q.trim()) {
      setHits([]);
      return;
    }
    let cancelled = false;
    const t = setTimeout(async () => {
      try {
        const r = await clientApi.searchCustomers(q.trim());
        if (!cancelled) {
          setHits(r);
          setActive(0);
        }
      } catch {
        // ignore
      }
    }, 180);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [q]);

  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  function pick(h: CustomerSearchHit) {
    setOpen(false);
    setQ("");
    router.push(`/c/${h.customer_id}`);
  }

  return (
    <div ref={root} className="relative w-[260px] sm:w-[320px]">
      <input
        value={q}
        onFocus={() => setOpen(true)}
        onChange={(e) => {
          setQ(e.target.value);
          setOpen(true);
        }}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") {
            e.preventDefault();
            setActive((a) => Math.min(a + 1, hits.length - 1));
          } else if (e.key === "ArrowUp") {
            e.preventDefault();
            setActive((a) => Math.max(a - 1, 0));
          } else if (e.key === "Enter" && hits[active]) {
            e.preventDefault();
            pick(hits[active]);
          } else if (e.key === "Escape") {
            setOpen(false);
          }
        }}
        placeholder="Search customer by name or ID"
        className="h-9 w-full border-b border-ink bg-transparent px-1 text-sm placeholder:text-mute focus:outline-none"
        aria-label="Search customer"
      />
      {open && hits.length > 0 && (
        <ul
          role="listbox"
          className="absolute right-0 top-[calc(100%+4px)] z-30 max-h-[320px] w-full overflow-y-auto border border-ink bg-cream shadow-[0_4px_24px_rgba(26,26,26,0.08)]"
        >
          {hits.map((h, i) => (
            <li
              key={h.customer_id}
              role="option"
              aria-selected={i === active}
              onMouseEnter={() => setActive(i)}
              onClick={() => pick(h)}
              className={`cursor-pointer px-3 py-2 text-sm ${
                i === active ? "bg-ink text-cream" : "text-ink hover:bg-bone"
              }`}
            >
              <div className="flex items-baseline justify-between gap-3">
                <div className="truncate font-display text-[15px]">{h.surname}</div>
                <div
                  className={`font-mono text-[11px] tabular ${
                    i === active ? "text-cream/80" : "text-mute"
                  }`}
                >
                  #{h.customer_id}
                </div>
              </div>
              <div
                className={`text-[11px] uppercase tracking-[0.14em] ${
                  i === active ? "text-cream/70" : "text-mute"
                }`}
              >
                {h.geography} / age {h.age}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
