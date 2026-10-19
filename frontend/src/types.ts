// Thin view-model aliases. Source of truth is src/services/api-types.ts (npm run gen:types).
// After generating, replace these with: import type { components } from "./services/api-types";
export type Decision = "PASS" | "FAIL" | "HUMAN" | "RETAKE";
export interface Category { id: string; display_name?: string; alpha_options?: number[]; default_alpha?: number; alpha_fail_options?: number[]; alpha_fail_default?: number; sample_count?: number; n_calib?: number }
export interface Sample { id?: string; sample_id?: string; category: string; display_name?: string; image_url?: string; ground_truth?: string; attribution?: string; role?: string | null; demo_role?: string | null }
export interface TraceStep { step: number; tool: string; reason: string; status: string; counts_toward_budget: boolean; duration_ms: number; overlay_url?: string | null }
export interface Inspection {
  inspection_id: string; engine: "live" | "mock"; category: string;
  snapshot: { model_version: string; bank_version: string; calibration_version: string; policy_version: string };
  decision: Decision; decision_path: string; suggested_decision: "PASS" | "FAIL" | null;
  anomaly_score: number | null;
  thresholds: { t_pass: number | null; t_fail: number | null; alpha_pass: number; alpha_fail: number; n_calib: number };
  calibration: { status: string; claim: string | null; assumptions: string };
  quality_gate: { status: string; reason: string };
  alignment: { quality: number; q_min: number; method: string } | null;
  measurements: { residual_mean: number; r_low: number; r_high: number; area_pct: number | null; elongation: number | null; shape_class: string | null; rois_checked: number; rois_total: number } | null;
  assets: { input_standardized: string; reference_aligned: string | null; anomaly_heatmap: string | null; contour_mask: string | null };
  execution_trace: TraceStep[]; steps_used: number; step_budget: number;
  timing_ms: { total: number };
  grounded_report: string;
  counterfactuals: { alpha_fail: number; t_fail: number; decision: Decision; needs_rerun: boolean }[];
  experimental_shadow: null | { shadow_version: string; score: number; shadow_t_pass: number; shadow_t_fail: number; note: string };
}
