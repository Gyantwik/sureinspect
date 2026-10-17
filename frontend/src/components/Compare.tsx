import { useState } from "react";
import type { Inspection } from "../types";
import { useAsset } from "../services/useAsset";

function Layer({ url, opacity = 1 }: { url?: string | null; opacity?: number }) {
  const { src } = useAsset(url);
  return src ? <img src={src} alt="" style={{ opacity }} className="absolute inset-0 h-full w-full object-contain" /> : null;
}

export default function Compare({ r }: { r: Inspection | null }) {
  const [base, setBase] = useState<"input" | "reference">("input");
  const [op, setOp] = useState(0.55);
  const baseUrl = base === "input" ? r?.assets.input_standardized : r?.assets.reference_aligned;
  const probe = useAsset(baseUrl);
  const hasRef = !!r?.assets.reference_aligned;
  return (
    <section className="panel p-5" aria-label="Compare">
      <div className="flex items-center justify-between mb-4">
        <h2 className="font-display text-2xl">Compare</h2>
        {r && (
          <div className="flex rounded-lg border border-line overflow-hidden text-xs">
            {(["input", "reference"] as const).map((b) => (
              <button key={b} disabled={b === "reference" && !hasRef} onClick={() => setBase(b)}
                className={`px-3 py-1.5 capitalize disabled:opacity-40 ${base === b ? "bg-ink text-paper" : "bg-white"}`}>{b}</button>
            ))}
          </div>
        )}
      </div>
      <div className="relative aspect-[4/3] w-full rounded-lg bg-[#e9e6df] overflow-hidden grid place-items-center">
        {!r && <span className="text-sm text-muted">No image yet</span>}
        {r && probe.failed && <span className="text-sm text-muted">Asset expired or unavailable</span>}
        {r && !probe.failed && <><Layer url={baseUrl} /><Layer url={r.assets.anomaly_heatmap} opacity={op} /></>}
      </div>
      {r && (
        <label className="mt-4 flex items-center gap-3 text-xs text-muted">
          <span className="label whitespace-nowrap">Heatmap</span>
          <input type="range" min={0} max={1} step={0.05} value={op} onChange={(e) => setOp(+e.target.value)} className="w-full accent-[#0e5a4c]" aria-label="Heatmap opacity" />
          <span className="font-mono w-9 text-right">{Math.round(op * 100)}%</span>
        </label>
      )}
    </section>
  );
}
