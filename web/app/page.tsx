import { redirect } from "next/navigation";

import { api } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function Home() {
  // Pick a featured customer; fall back to a known default if API is offline.
  // The default surname "Hargrave" is the first row in churn.csv (CustomerId 15634602).
  let target = 15634602;
  try {
    const hits = await api.searchCustomers("Hargrave", 1);
    if (hits[0]) target = hits[0].customer_id;
  } catch {
    // FastAPI not up yet -- use the static default.
  }
  redirect(`/c/${target}`);
}
