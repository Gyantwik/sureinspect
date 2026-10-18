import type { Category, Inspection, Sample } from "../types";

const KEY = "si_session";
let token: string | null = typeof sessionStorage !== "undefined" ? sessionStorage.getItem(KEY) : null;

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public requestId?: string) { super(message); }
}

async function check(r: Response): Promise<Response> {
  if (r.ok) return r;
  let code = `HTTP_${r.status}`, msg = r.statusText, rid: string | undefined;
  try { const j = await r.json(); code = j.error?.code ?? code; msg = j.error?.message ?? msg; rid = j.error?.request_id; } catch { /* non-JSON */ }
  throw new ApiError(r.status, code, msg, rid);
}

export async function ensureSession(force = false): Promise<string> {
  if (token && !force) return token;
  const j = await (await check(await fetch("/api/session", { method: "POST" }))).json();
  token = j.session_token as string;
  if (typeof sessionStorage !== "undefined") {
    sessionStorage.setItem(KEY, token);
  }
  return token;
}

async function authed(path: string, init: RequestInit = {}): Promise<Response> {
  const send = (t: string) => fetch(path, { ...init, headers: { ...(init.headers ?? {}), Authorization: `Bearer ${t}` } });
  let r = await send(await ensureSession());
  if (r.status === 401) r = await send(await ensureSession(true)); // expired: one transparent renewal
  return check(r);
}

const list = <T,>(j: any, key: string): T[] => (Array.isArray(j) ? j : j?.[key] ?? []);

export const getHealth = async () => (await check(await fetch("/api/health"))).json();
export const getCategories = async () => list<Category>(await (await check(await fetch("/api/categories"))).json(), "categories");
export const getSamples = async (c: string) => list<Sample>(await (await check(await fetch(`/api/samples?category=${encodeURIComponent(c)}`))).json(), "samples");

export interface InspectArgs { category: string; file?: File; sampleId?: string; alphaFail?: number; blur: boolean; glare: boolean; lighting: boolean }
export async function inspect(a: InspectArgs): Promise<Inspection> {
  const f = new FormData();
  if (a.file) f.append("file", a.file); else if (a.sampleId) f.append("sample_id", a.sampleId);
  f.append("category", a.category);
  if (a.alphaFail != null) f.append("alpha_fail", String(a.alphaFail));
  // Field names: confirm against contracts/openapi.json
  f.append("simulate_blur", String(a.blur));
  f.append("simulate_glare", String(a.glare));
  f.append("simulate_darken", String(a.lighting));
  return (await authed("/api/inspect", { method: "POST", body: f })).json();
}

export async function assetObjectUrl(url: string): Promise<string> {
  return URL.createObjectURL(await (await authed(url)).blob());
}

export async function sendFeedback(inspectionId: string, label: "NORMAL" | "DEFECT") {
  const b = crypto.getRandomValues(new Uint8Array(12));
  const key = Array.from(b, (x) => x.toString(16).padStart(2, "0")).join(""); // randomUUID needs a secure context
  await authed("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ inspection_id: inspectionId, user_label: label, idempotency_key: key }) });
}

export async function downloadTrace(id: string) {
  const j = await (await authed(`/api/trace/${id}`)).json();
  const u = URL.createObjectURL(new Blob([JSON.stringify(j, null, 2)], { type: "application/json" }));
  Object.assign(document.createElement("a"), { href: u, download: `trace-${id}.json` }).click();
  URL.revokeObjectURL(u);
}
