import { Play, Upload } from "lucide-react";
import type { Category, Sample } from "../types";

export interface Form { category: string; sampleId: string; file: File | null; alpha: number | undefined; blur: boolean; glare: boolean; lighting: boolean }
interface Props { cats: Category[]; samples: Sample[]; form: Form; set: (p: Partial<Form>) => void; busy: boolean; onRun: () => void }

export default function ControlPanel({ cats, samples, form, set, busy, onRun }: Props) {
  const cat = cats.find((c) => c.id === form.category);
  const canRun = !busy && !!form.category && (!!form.file || !!form.sampleId);
  return (
    <section className="panel p-5 flex flex-col gap-5" aria-label="Controls">
      <h2 className="font-display text-2xl">Inspection</h2>

      <label className="flex flex-col gap-1.5">
        <span className="label">Category</span>
        <select className="rounded-lg border border-line bg-white px-3 py-2 text-sm" value={form.category} onChange={(e) => set({ category: e.target.value, sampleId: "" })}>
          {cats.map((c) => <option key={c.id} value={c.id}>{c.display_name ?? c.id}</option>)}
        </select>
      </label>

      <div className="flex flex-col gap-1.5">
        <span className="label">Image source</span>
        <select className="rounded-lg border border-line bg-white px-3 py-2 text-sm" value={form.sampleId} onChange={(e) => set({ sampleId: e.target.value, file: null })} aria-label="Sample">
          <option value="">Choose a curated sample…</option>
          {samples.map((s) => {
            const sid = s.id ?? s.sample_id ?? "";
            const srole = s.role ?? s.demo_role;
            return <option key={sid} value={sid}>{sid}{srole ? ` · ${srole}` : ""}</option>;
          })}
        </select>
        <label className="flex items-center gap-2 rounded-lg border border-dashed border-line px-3 py-2 text-sm text-muted cursor-pointer hover:border-accent focus-within:outline-2">
          <Upload size={16} />
          <span className="truncate">{form.file ? form.file.name : "or upload an image"}</span>
          <input type="file" accept="image/*" className="sr-only" onChange={(e) => set({ file: e.target.files?.[0] ?? null, sampleId: "" })} />
        </label>
      </div>

      {(cat?.alpha_fail_options ?? cat?.alpha_options) && (
        <div className="flex flex-col gap-1.5">
          <span className="label">Fail risk α</span>
          <div className="flex gap-2" role="radiogroup" aria-label="Alpha">
            {(cat.alpha_fail_options ?? cat.alpha_options ?? []).map((a) => (
              <button key={a} role="radio" aria-checked={form.alpha === a} onClick={() => set({ alpha: a })}
                className={`flex-1 rounded-lg border px-2 py-1.5 text-sm font-mono ${form.alpha === a ? "bg-ink text-paper border-ink" : "bg-white border-line"}`}>{a}</button>
            ))}
          </div>
        </div>
      )}

      <fieldset className="flex flex-col gap-2">
        <legend className="label mb-1">Simulate degradation</legend>
        {([["blur", "Blur"], ["glare", "Glare"], ["lighting", "Lighting shift"]] as const).map(([k, t]) => (
          <label key={k} className="flex items-center gap-2 text-sm">
            <input type="checkbox" className="accent-[#0e5a4c]" checked={form[k]} onChange={(e) => set({ [k]: e.target.checked } as Partial<Form>)} /> {t}
          </label>
        ))}
      </fieldset>

      <button disabled={!canRun} onClick={onRun}
        className="flex items-center justify-center gap-2 rounded-lg bg-accent text-white py-2.5 text-sm font-medium disabled:opacity-40">
        <Play size={15} /> {busy ? "Inspecting…" : "Run inspection"}
      </button>

      <p className="text-xs leading-relaxed text-muted">
        Uploads are processed for inspection and audit. Local cache is short-lived; audit records become eligible for automatic deletion after 14 days. Do not upload confidential images.
      </p>
    </section>
  );
}
