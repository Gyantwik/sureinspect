import { useState } from "react";
import { Check, Flag } from "lucide-react";
import type { Decision, Inspection } from "../types";
import { sendFeedback } from "../services/api";

const tone: Record<Decision, string> = { PASS: "text-pass", FAIL: "text-fail", HUMAN: "text-human", RETAKE: "text-retake" };
const word: Record<Decision, string> = { PASS: "Pass", FAIL: "Fail", HUMAN: "Human review", RETAKE: "Retake" };
const next: Record<Decision, string> = { PASS: "Continue the line and retain this trace for audit.", FAIL: "Quarantine the item and request a defect review.", HUMAN: "Send the item and evidence trace to a trained reviewer.", RETAKE: "Capture a new image under the requested quality conditions." };
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
  const [sending, setSending] = useState(false);
  if (!r) return <section className="panel p-5"><h2 className="font-display text-2xl">Decision</h2><p className="text-sm text-muted mt-3">The official decision appears here.</p></section>;
  const m = r.measurements;
  const fb = async (l: "NORMAL" | "DEFECT") => { setSending(true); try { await sendFeedback(r.inspection_id, l); setSent(l); } catch { setSent("error"); } finally { setSending(false); } };
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
      <Row k="Recovered geometry" v={r.alignment?.scale != null ? `${f(r.alignment.scale, 3)}× scale · ${f(r.alignment.overlap, 2)} overlap` : "not measured"} />

      <p className="text-xs text-muted mt-4 leading-relaxed"><span className="label block mb-1">Calibration · {r.calibration.status.replace(/_/g, " ")}</span>{r.calibration.assumptions}</p>
      <p className="text-sm leading-relaxed mt-4">{r.grounded_report}</p>
      <div className="mt-4 rounded-lg border border-line bg-white p-3 text-sm">
        <span className="label block mb-1">Next action · {r.next_action?.action?.replace(/_/g, " ") ?? r.decision}</span>
        <p>{r.next_action?.reason ?? next[r.decision]}</p>
        <p className="mt-2 text-xs text-muted">
          {r.next_action?.requires_human_confirmation ? "Human confirmation required" : "No confirmation required"}
          {" · "}Recommendation only · No external actuation
        </p>
      </div>

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
        <button type="button" disabled={sending || r.decision === "RETAKE"} onClick={() => fb("NORMAL")} className="flex items-center gap-1 rounded-lg border border-line bg-white px-3 py-2 text-xs disabled:opacity-40"><Check aria-hidden="true" size={13} />Normal</button>
        <button type="button" disabled={sending || r.decision === "RETAKE"} onClick={() => fb("DEFECT")} className="flex items-center gap-1 rounded-lg border border-line bg-white px-3 py-2 text-xs disabled:opacity-40"><Flag aria-hidden="true" size={13} />Defect</button>
      </div>
      {sent && <p role="status" className="text-xs text-muted mt-2">{sent === "error" ? "Feedback failed; please retry." : `Logged for audit only: ${sent}`}</p>}

      <p className="font-mono text-[10px] text-muted mt-5 break-all">{r.snapshot.model_version} · {r.snapshot.bank_version} · {r.snapshot.calibration_version} · {r.snapshot.policy_version}</p>
    </section>
  );
}
