import { NextRequest, NextResponse } from "next/server";

const FASTAPI_URL = process.env.FASTAPI_URL;

export const runtime = "nodejs";

async function forward(req: NextRequest, params: { path: string[] }) {
  // No backend configured: the client-side api wrapper will fall back to
  // fixtures. Return 503 immediately so it doesn't wait for a TCP timeout.
  if (!FASTAPI_URL) {
    return new NextResponse(
      JSON.stringify({ error: "FASTAPI_URL not configured; client should fall back to fixtures." }),
      { status: 503, headers: { "Content-Type": "application/json" } }
    );
  }

  const path = params.path.join("/");
  const url = new URL(`${FASTAPI_URL}/${path}`);
  for (const [k, v] of req.nextUrl.searchParams.entries()) {
    url.searchParams.set(k, v);
  }

  const init: RequestInit = {
    method: req.method,
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
  };
  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = await req.text();
  }

  const upstream = await fetch(url.toString(), init);
  const body = await upstream.text();
  return new NextResponse(body, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") || "application/json",
    },
  });
}

export async function GET(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return forward(req, await ctx.params);
}
export async function POST(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return forward(req, await ctx.params);
}
