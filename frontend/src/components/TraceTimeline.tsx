import { Download } from "lucide-react";
import type { Inspection } from "../types";
import { downloadTrace } from "../services/api";

export default function TraceTimeline({ r }: { r: Inspection | null }) {
  return (
    <section className="panel p-5" aria-label="Trace">
      <div className="flex items-center justify-between mb-4">
        <h2 className="font-display text-2xl">Trace</h2>
        {r && (
          <div className="flex items-center gap-3">
            <span className="label">Budget {r.steps_used}/{r.step_budget} · {Math.round(r.timing_ms.total)} ms</span>
            <button className="text-muted hover:text-ink" title="Download trace JSON" aria-label="Download trace" onClick={() => downloadTrace(r.inspection_id)}><Download size={16} /></button>
          </div>
        )}
      </div>
      {!r ? <p className="text-sm text-muted">Run an inspection to see each tool step the agent took.</p> : (
        <ol className="relative border-l border-line ml-2 flex flex-col gap-4">
          {r.execution_trace.map((s) => (
            <li key={s.step} className="pl-5 relative">
              <span className={`absolute -left-[5px] top-1.5 h-2.5 w-2.5 rounded-full ${s.status.toLowerCase().includes("fail") ? "bg-fail" : "bg-accent"}`} />
              <div className="flex items-baseline justify-between gap-3">
                <span className="font-mono text-xs">{s.step}. {s.tool}</span>
                <span className="font-mono text-xs text-muted">{s.duration_ms.toFixed(0)} ms{s.counts_toward_budget ? "" : " · free"}</span>
              </div>
              <p className="text-sm text-muted mt-0.5">{s.reason}</p>
              <span className="label">{s.status}</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
