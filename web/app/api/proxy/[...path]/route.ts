import { NextRequest, NextResponse } from "next/server";

const FASTAPI_URL = process.env.FASTAPI_URL;
const IS_PROD = process.env.NODE_ENV === "production";
const UPSTREAM_TIMEOUT_MS = 30_000;

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
  let url: URL;
  try {
    url = new URL(`${FASTAPI_URL.replace(/\/$/, "")}/${path}`);
  } catch (err) {
    console.error("[proxy] Invalid FASTAPI_URL:", err);
    return new NextResponse(
      JSON.stringify({
        error: "Invalid backend configuration",
        ...(IS_PROD
          ? {}
          : { message: err instanceof Error ? err.message : String(err), fastapi_url: FASTAPI_URL }),
      }),
      { status: 502, headers: { "Content-Type": "application/json" } }
    );
  }

  for (const [k, v] of req.nextUrl.searchParams.entries()) {
    url.searchParams.set(k, v);
  }

  const init: RequestInit = {
    method: req.method,
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
  };
  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = await req.text();
  }

  try {
    const upstream = await fetch(url.toString(), init);
    const body = await upstream.text();
    return new NextResponse(body, {
      status: upstream.status,
      headers: {
        "Content-Type":
          upstream.headers.get("content-type") || "application/json",
      },
    });
  } catch (err) {
    // Log full detail server-side; redact in production responses so we
    // don't leak the upstream URL or transport errors to clients.
    console.error("[proxy] upstream fetch failed:", url.toString(), err);
    const isAbort = err instanceof DOMException && err.name === "TimeoutError";
    return new NextResponse(
      JSON.stringify({
        error: isAbort ? "Upstream timed out" : "Upstream fetch failed",
        ...(IS_PROD
          ? {}
          : {
              upstream: url.toString(),
              message: err instanceof Error ? err.message : String(err),
              cause:
                err instanceof Error && "cause" in err
                  ? String((err as Error & { cause?: unknown }).cause)
                  : undefined,
            }),
      }),
      { status: isAbort ? 504 : 502, headers: { "Content-Type": "application/json" } }
    );
  }
}

export async function GET(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return forward(req, await ctx.params);
}
export async function POST(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return forward(req, await ctx.params);
}
