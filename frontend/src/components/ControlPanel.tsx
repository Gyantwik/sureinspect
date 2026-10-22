import { useState, useRef } from "react";
import { Camera, Play, Upload, X, Smartphone } from "lucide-react";
import type { Category, Sample } from "../types";
import CameraCaptureModal from "./CameraCaptureModal";

export interface Form {
  category: string;
  sampleId: string;
  file: File | null;
  alpha: number | undefined;
  blur: boolean;
  glare: boolean;
  lighting: boolean;
}

interface Props {
  cats: Category[];
  samples: Sample[];
  form: Form;
  set: (p: Partial<Form>) => void;
  busy: boolean;
  onRun: () => void;
}

export default function ControlPanel({ cats, samples, form, set, busy, onRun }: Props) {
  const [cameraModalOpen, setCameraModalOpen] = useState(false);
  const nativeCameraInputRef = useRef<HTMLInputElement | null>(null);

  const cat = cats.find((c) => c.id === form.category);
  const canRun = !busy && !!form.category && (!!form.file || !!form.sampleId);

  // Handle camera capture result
  const handlePhotoCaptured = (file: File) => {
    set({ file, sampleId: "" });
  };

  // Clear current custom file
  const handleClearFile = () => {
    set({ file: null });
  };

  return (
    <>
      <section className="panel p-4 sm:p-5 flex flex-col gap-4 sm:gap-5" aria-label="Controls">
        <div className="flex items-center justify-between">
          <h2 className="font-display text-xl sm:text-2xl">Inspection</h2>
          <span className="text-[11px] font-mono uppercase tracking-wider text-muted bg-surface/60 px-2 py-0.5 rounded border border-line">
            Config
          </span>
        </div>

        {/* Category Selector */}
        <label className="flex flex-col gap-1.5">
          <span className="label">Category</span>
          <select
            className="rounded-lg border border-line bg-white px-3 py-2.5 text-base sm:text-sm text-fg"
            value={form.category}
            onChange={(e) => set({ category: e.target.value, sampleId: "" })}
          >
            {cats.map((c) => (
              <option key={c.id} value={c.id}>
                {c.display_name ?? c.id}
              </option>
            ))}
          </select>
        </label>

        {/* Image Source Selection */}
        <div className="flex flex-col gap-2">
          <span className="label">Image Source</span>

          {/* Curated Sample Dropdown */}
          <select
            className="rounded-lg border border-line bg-white px-3 py-2.5 text-base sm:text-sm text-fg"
            value={form.sampleId}
            onChange={(e) => set({ sampleId: e.target.value, file: null })}
            aria-label="Sample"
          >
            <option value="">Choose a curated sample…</option>
            {samples.map((s) => {
              const sid = s.id ?? s.sample_id ?? "";
              const sname = s.display_name || sid;
              const srole = s.role ?? s.demo_role;
              return (
                <option key={sid} value={sid}>
                  {sname}{srole ? ` [${srole}]` : ""}
                </option>
              );
            })}
          </select>

          {/* Direct Camera & File Upload Actions */}
          <div className="grid grid-cols-2 gap-2 mt-1">
            {/* Live Camera Viewfinder Modal */}
            <button
              type="button"
              onClick={() => setCameraModalOpen(true)}
              className="flex items-center justify-center gap-1.5 rounded-lg border border-line bg-white hover:bg-surface/50 active:bg-surface px-3 py-2.5 text-xs sm:text-sm font-medium text-fg transition cursor-pointer"
              title="Open live camera viewfinder"
            >
              <Camera size={16} className="text-accent" />
              <span>Live Camera</span>
            </button>

            {/* Native Mobile Camera Snap */}
            <button
              type="button"
              onClick={() => nativeCameraInputRef.current?.click()}
              className="flex items-center justify-center gap-1.5 rounded-lg border border-line bg-white hover:bg-surface/50 active:bg-surface px-3 py-2.5 text-xs sm:text-sm font-medium text-fg transition cursor-pointer"
              title="Snap photo with native phone camera"
            >
              <Smartphone size={16} className="text-accent" />
              <span>Take Photo</span>
            </button>
            <input
              ref={nativeCameraInputRef}
              type="file"
              accept="image/png,image/jpeg"
              capture="environment"
              className="sr-only"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handlePhotoCaptured(f);
              }}
            />
          </div>

          {/* Regular Upload Dropzone */}
          <label className="flex items-center justify-center gap-2 rounded-lg border border-dashed border-line px-3 py-2.5 text-xs sm:text-sm text-muted cursor-pointer hover:border-accent hover:text-fg focus-within:outline-2 transition">
            <Upload size={15} />
            <span className="truncate">Or upload an image file</span>
            <input
              type="file"
              accept="image/png,image/jpeg"
              className="sr-only"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handlePhotoCaptured(f);
              }}
            />
          </label>

          {/* Attached Image Preview Pill */}
          {form.file && (
            <div className="flex items-center justify-between gap-2 p-2 rounded-lg bg-accent/5 border border-accent/20 text-xs">
              <div className="flex items-center gap-2 min-w-0">
                <span className="inline-block w-2 h-2 rounded-full bg-accent shrink-0" />
                <span className="truncate font-medium text-accent font-mono">{form.file.name}</span>
                <span className="text-muted shrink-0">({(form.file.size / 1024).toFixed(0)} KB)</span>
              </div>
              <button
                type="button"
                onClick={handleClearFile}
                className="p-1 rounded hover:bg-black/5 text-muted hover:text-fg transition shrink-0"
                title="Remove image"
              >
                <X size={14} />
              </button>
            </div>
          )}
        </div>

        {/* Conformal Risk Budget Selector */}
        {(cat?.alpha_fail_options ?? cat?.alpha_options) && (
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between">
              <span className="label">Fail risk limit α</span>
              <span className="text-[11px] font-mono text-muted">
                {form.alpha !== undefined ? `${((1 - form.alpha) * 100).toFixed(0)}% guarantee` : ""}
              </span>
            </div>
            <div className="flex gap-1.5 sm:gap-2" role="radiogroup" aria-label="Alpha">
              {(cat.alpha_fail_options ?? cat.alpha_options ?? []).map((a) => (
                <button
                  type="button"
                  key={a}
                  role="radio"
                  aria-checked={form.alpha === a}
                  onClick={() => set({ alpha: a })}
                  className={`flex-1 rounded-lg border py-2 text-xs sm:text-sm font-mono font-medium transition ${
                    form.alpha === a
                      ? "bg-ink text-paper border-ink shadow-xs"
                      : "bg-white border-line text-muted hover:text-fg"
                  }`}
                >
                  {a}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Optical Degradation Simulation Toggles */}
        <fieldset className="flex flex-col gap-2 pt-1 border-t border-line/60">
          <legend className="label mb-1">Simulate Optical Degradation</legend>
          {([
            ["blur", "Defocus Blur (Admission Reject)"],
            ["glare", "Specular Glare (Admission Reject)"],
            ["lighting", "Lighting Shift (Triage Review)"],
          ] as const).map(([k, t]) => (
            <label key={k} className="flex items-center gap-2.5 text-xs sm:text-sm text-fg cursor-pointer select-none">
              <input
                type="checkbox"
                className="w-4 h-4 rounded accent-[#0e5a4c] cursor-pointer"
                checked={form[k]}
                onChange={(e) => set({ [k]: e.target.checked } as Partial<Form>)}
              />
              <span>{t}</span>
            </label>
          ))}
        </fieldset>

        {/* Run Button */}
        <button
          type="button"
          disabled={!canRun}
          aria-busy={busy}
          onClick={onRun}
          className="flex items-center justify-center gap-2 rounded-xl bg-accent hover:bg-[#127260] active:scale-[0.98] text-white py-3 text-sm font-medium disabled:opacity-40 transition shadow-xs cursor-pointer"
        >
          <Play size={16} className={busy ? "animate-pulse" : ""} />
          <span>{busy ? "Running OpenCV Inspection…" : "Run Inspection"}</span>
        </button>

        <p className="text-[11px] leading-relaxed text-muted">
          Captured or uploaded images are processed via OpenCV 5 and PatchCore. Assets expire after 30 idle minutes.
        </p>
      </section>

      {/* Live Camera Viewfinder Modal */}
      <CameraCaptureModal
        isOpen={cameraModalOpen}
        onClose={() => setCameraModalOpen(false)}
        onCapture={handlePhotoCaptured}
      />
    </>
  );
}
