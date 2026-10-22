import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { ApiError, getCategories, getHealth, getSamples, inspect, resetSession } from "./services/api";
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
  const [resetting, setResetting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const set = useCallback((p: Partial<Form>) => setForm((f) => ({ ...f, ...p })), []);

  const loadInitialData = useCallback(async () => {
    setError(null);
    try {
      const [health, categories] = await Promise.all([getHealth(), getCategories()]);
      setEngine(health.engine);
      setCats(categories);
      if (categories[0]) set({ category: categories[0].id, alpha: categories[0].alpha_fail_default ?? categories[0].default_alpha });
    } catch {
      setEngine(null);
      setError("Backend or category data is unavailable. Start the API, then retry.");
    }
  }, [set]);

  useEffect(() => { void loadInitialData(); }, [loadInitialData]);

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

  const handleReset = async () => {
    setResetting(true);
    setError(null);
    try {
      const res = await resetSession();
      setStatusMsg(res.message || "Session reset successfully.");
      setTimeout(() => setStatusMsg(null), 4000);
    } catch {
      setError("Failed to reset session.");
    } finally {
      setResetting(false);
    }
  };

  const engineLabel = engine === "mock" ? "MOCK ENGINE (Synthetic Evaluation)" : engine ? "LIVE OPENCV INFERENCE" : "ENGINE STATUS UNKNOWN";
  const engineTone = engine === "mock" ? "bg-human/15 text-human" : engine ? "bg-pass/15 text-pass" : "bg-retake/15 text-retake";
  return (
    <div className="min-h-screen flex flex-col justify-between">
      <a href="#main-content" className="skip-link">Skip to inspection controls</a>
      <div>
        <header className="border-b border-line">
          <div className="mx-auto max-w-[1400px] px-6 h-14 flex items-center justify-between">
            <div className="font-display text-2xl tracking-tight">SureInspect</div>
            <div className="flex items-center gap-3">
              <span className={`rounded-full px-3 py-1 text-[11px] font-mono tracking-wider ${engineTone}`}>
                {engineLabel}
              </span>
              <button
                type="button"
                onClick={handleReset}
                disabled={busy || resetting}
                title="Reset session shadow state"
                className="flex items-center gap-1.5 rounded border border-line bg-surface/50 px-2.5 py-1 text-xs text-muted hover:text-fg hover:border-line-hover transition-colors disabled:opacity-50 cursor-pointer"
              >
                <RefreshCw aria-hidden="true" size={12} className={resetting ? "animate-spin" : ""} />
                <span>Reset Session</span>
              </button>
            </div>
          </div>
        </header>

        <main id="main-content" className="mx-auto max-w-[1400px] px-4 sm:px-6 py-6">
          {error && (
            <div role="alert" className="mb-5 flex items-start gap-2 rounded-lg border border-fail/40 bg-fail/5 px-4 py-3 text-sm text-fail">
              <AlertTriangle aria-hidden="true" size={16} className="mt-0.5" />
              <span className="flex-1">{error}</span>
              <button type="button" className="rounded border border-fail/40 px-3 py-2 font-medium" onClick={() => void loadInitialData()}>Retry</button>
            </div>
          )}
          {statusMsg && (
            <div role="status" className="mb-5 flex items-start gap-2 rounded-lg border border-pass/40 bg-pass/5 px-4 py-3 text-sm text-pass">
              {statusMsg}
            </div>
          )}
          <div className="grid gap-5 lg:grid-cols-[300px_minmax(0,1fr)_340px] items-start">
            <ControlPanel cats={cats} samples={samples} form={form} set={set} busy={busy} onRun={run} />
            <div className="flex flex-col gap-5"><Compare r={result} /><TraceTimeline r={result} /></div>
            <DecisionCard r={result} />
          </div>
        </main>
      </div>

      <footer className="mt-12 border-t border-line py-4 text-center text-xs text-muted">
        Images and audit feedback remain only in the bounded in-memory session cache and expire after 30 idle minutes. Do not upload confidential images.
      </footer>
    </div>
  );
}

