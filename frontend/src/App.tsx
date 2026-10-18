import { useCallback, useEffect, useState } from "react";
import { AlertTriangle } from "lucide-react";
import { ApiError, getCategories, getHealth, getSamples, inspect } from "./services/api";
import type { Category, Inspection, Sample } from "./types";
import ControlPanel, { type Form } from "./components/ControlPanel";
import TraceTimeline from "./components/TraceTimeline";
import Compare from "./components/Compare";
import DecisionCard from "./components/DecisionCard";

const blank: Form = { category: "", sampleId: "", file: null, alpha: undefined, blur: false, glare: false, lighting: false };

export default function App() {
  const [cats, setCats] = useState<Category[]>([]);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [form, setForm] = useState<Form>(blank);
  const [result, setResult] = useState<Inspection | null>(null);
  const [engine, setEngine] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = useCallback((p: Partial<Form>) => setForm((f) => ({ ...f, ...p })), []);

  useEffect(() => {
    getHealth().then((h) => setEngine(h.engine)).catch(() => setError("Backend unreachable. Start it with: uvicorn backend.main:app"));
    getCategories().then((c) => { setCats(c); if (c[0]) set({ category: c[0].id, alpha: c[0].alpha_fail_default ?? c[0].default_alpha }); }).catch(() => {});
  }, [set]);

  useEffect(() => { if (form.category) getSamples(form.category).then(setSamples).catch(() => setSamples([])); }, [form.category]);

  const run = async () => {
    setBusy(true); setError(null);
    try {
      const r = await inspect({ category: form.category, file: form.file ?? undefined, sampleId: form.sampleId || undefined, alphaFail: form.alpha, blur: form.blur, glare: form.glare, lighting: form.lighting });
      setResult(r); setEngine(r.engine);
    } catch (e) {
      setError(e instanceof ApiError ? `${e.code}: ${e.message}${e.requestId ? ` (ref ${e.requestId})` : ""}` : "Inspection failed.");
    } finally { setBusy(false); }
  };

  const mock = engine === "mock";
  return (
    <div className="min-h-screen">
      <header className="border-b border-line">
        <div className="mx-auto max-w-[1400px] px-6 h-14 flex items-center justify-between">
          <div className="font-display text-2xl tracking-tight">SureInspect</div>
          {engine ? (
            <span className={`rounded-full px-3 py-1 text-[11px] font-mono tracking-wider ${mock ? "bg-human/15 text-human" : "bg-pass/15 text-pass"}`}>
              {mock ? "MOCK ENGINE" : "REAL ENGINE"}
            </span>
          ) : (
            <span className="rounded-full px-3 py-1 text-[11px] font-mono tracking-wider bg-line/40 text-muted">
              CONNECTING...
            </span>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-[1400px] px-6 py-6">
        {error && (
          <div role="alert" className="mb-5 flex items-start gap-2 rounded-lg border border-fail/40 bg-fail/5 px-4 py-3 text-sm text-fail">
            <AlertTriangle size={16} className="mt-0.5" /> {error}
          </div>
        )}
        <div className="grid gap-5 lg:grid-cols-[300px_minmax(0,1fr)_340px] items-start">
          <ControlPanel cats={cats} samples={samples} form={form} set={set} busy={busy} onRun={run} />
          <div className="flex flex-col gap-5"><Compare r={result} /><TraceTimeline r={result} /></div>
          <DecisionCard r={result} />
        </div>
      </main>

      <footer className="mt-12 border-t border-line py-6">
        <div className="mx-auto max-w-[1400px] px-6 text-center text-xs text-muted">
          <p>
            Uploads are processed for inspection and audit. Local cache is short-lived; audit records become eligible for automatic deletion after 14 days. Do not upload confidential images.
          </p>
        </div>
      </footer>
    </div>
  );
}
