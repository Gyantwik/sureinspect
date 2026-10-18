import { useState } from "react";
import { Check, Flag } from "lucide-react";
import type { Decision, Inspection } from "../types";
import { sendFeedback } from "../services/api";

const tone: Record<Decision, string> = { PASS: "text-pass", FAIL: "text-fail", HUMAN: "text-human", RETAKE: "text-retake" };
const word: Record<Decision, string> = { PASS: "Pass", FAIL: "Fail", HUMAN: "Human review", RETAKE: "Retake" };
const f = (n: number | null | undefined, d = 3) => (n == null ? "not measured" : n.toFixed(d));

function Row({ k, v }: { k: string; v: string }) {
  return <div className="flex justify-between gap-4 py-1.5 border-b border-line/70 text-sm"><span className="text-muted">{k}</span><span className="font-mono text-right">{v}</span></div>;
}

function ScoreBar({ r }: { r: Inspection }) {
  const { anomaly_score: s, thresholds: t } = r;
  if (s == null || t.t_pass == null || t.t_fail == null) return null;
  const max = Math.max(s, t.t_fail) * 1.25, pos = (v: number) => `${Math.min(100, (v / max) * 100)}%`;
  return (
    <div className="my-5" aria-label="Score against thresholds">
      <div className="relative h-2 rounded-full bg-line">
        <div className="absolute h-2 rounded-full bg-ink" style={{ width: pos(s) }} />
        {[t.t_pass, t.t_fail].map((v, i) => <span key={i} className="absolute -top-1 h-4 w-px bg-accent" style={{ left: pos(v) }} />)}
      </div>
      <div className="relative mt-2 h-4 font-mono text-[10px] text-muted">
        <span className="absolute -translate-x-1/2" style={{ left: pos(t.t_pass) }}>t_pass</span>
        <span className="absolute -translate-x-1/2" style={{ left: pos(t.t_fail) }}>t_fail</span>
      </div>
    </div>
  );
}

export default function DecisionCard({ r }: { r: Inspection | null }) {
  const [sent, setSent] = useState<string | null>(null);
  if (!r) return <section className="panel p-5"><h2 className="font-display text-2xl">Decision</h2><p className="text-sm text-muted mt-3">The official decision appears here.</p></section>;
  const m = r.measurements;
  const fb = async (l: "NORMAL" | "DEFECT") => { try { await sendFeedback(r.inspection_id, l); setSent(l); } catch { setSent("error"); } };
  return (
    <section className="panel p-5" aria-label="Decision" aria-live="polite">
      <span className="label">Official frozen decision</span>
      <div className={`font-display text-5xl mt-1 ${tone[r.decision]}`}>{word[r.decision]}</div>
      <div className="font-mono text-xs text-muted mt-1">{r.decision_path}{r.suggested_decision ? ` · suggests ${r.suggested_decision}` : ""}</div>
      <ScoreBar r={r} />

      <Row k="Anomaly score" v={f(r.anomaly_score)} />
      <Row k="t_pass / t_fail" v={`${f(r.thresholds.t_pass)} / ${f(r.thresholds.t_fail)}`} />
      <Row k="α_fail · n_calib" v={`${r.thresholds.alpha_fail} · ${r.thresholds.n_calib}`} />
      <Row k="Residual mean" v={m ? f(m.residual_mean) : "not measured"} />
      <Row k="Area %" v={m ? f(m.area_pct, 2) : "not measured"} />
      <Row k="Shape" v={m?.shape_class ?? "not measured"} />
      <Row k="Alignment" v={r.alignment ? `${f(r.alignment.quality, 2)} (min ${f(r.alignment.q_min, 2)})` : "not measured"} />

      <p className="text-xs text-muted mt-4 leading-relaxed"><span className="label block mb-1">Calibration · {r.calibration.status.replace(/_/g, " ")}</span>{r.calibration.assumptions}</p>
      <p className="text-sm leading-relaxed mt-4">{r.grounded_report}</p>

      {r.counterfactuals.length > 0 && (
        <div className="mt-4"><span className="label">Counterfactual α</span>
          {r.counterfactuals.map((c) => <Row key={c.alpha_fail} k={`α ${c.alpha_fail}`} v={`${c.decision}${c.needs_rerun ? " · rerun" : ""}`} />)}
        </div>
      )}

      {r.experimental_shadow && (
        <div className="mt-4 rounded-lg border border-dashed border-human p-3">
          <span className="label text-human">Experimental shadow</span>
          <p className="font-mono text-sm mt-1">score {f(r.experimental_shadow.score)}</p>
          <p className="text-xs text-muted mt-1">Experimental; no calibration claim; official decision unchanged.</p>
        </div>
      )}

      <div className="mt-5 flex items-center gap-2">
        <span className="label mr-auto">Ground truth?</span>
        <button onClick={() => fb("NORMAL")} className="flex items-center gap-1 rounded-lg border border-line bg-white px-3 py-1.5 text-xs"><Check size={13} />Normal</button>
        <button onClick={() => fb("DEFECT")} className="flex items-center gap-1 rounded-lg border border-line bg-white px-3 py-1.5 text-xs"><Flag size={13} />Defect</button>
      </div>
      {sent && <p className="text-xs text-muted mt-2">{sent === "error" ? "Feedback failed." : `Logged: ${sent}`}</p>}

      <p className="font-mono text-[10px] text-muted mt-5 break-all">{r.snapshot.model_version} · {r.snapshot.bank_version} · {r.snapshot.calibration_version} · {r.snapshot.policy_version}</p>
    </section>
  );
}
