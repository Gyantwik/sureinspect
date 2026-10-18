import { describe, it, expect } from "vitest";
import type { Decision, Inspection } from "../types";

describe("Decision Types and Path Logic", () => {
  it("validates all 4 primary decision enums", () => {
    const validDecisions: Decision[] = ["PASS", "FAIL", "HUMAN", "RETAKE"];
    expect(validDecisions).toHaveLength(4);
    expect(validDecisions).toContain("PASS");
    expect(validDecisions).toContain("FAIL");
    expect(validDecisions).toContain("HUMAN");
    expect(validDecisions).toContain("RETAKE");
  });

  it("handles valid inspection structure", () => {
    const mockInspection: Partial<Inspection> = {
      inspection_id: "insp_test_001",
      engine: "mock",
      category: "pcb1",
      decision: "PASS",
      decision_path: "FAST_PASS",
      anomaly_score: 0.12,
      thresholds: {
        t_pass: 0.20,
        t_fail: 0.50,
        alpha_pass: 0.20,
        alpha_fail: 0.05,
        n_calib: 300,
      },
    };

    expect(mockInspection.decision).toBe("PASS");
    expect(mockInspection.anomaly_score).toBeLessThan(mockInspection.thresholds!.t_pass!);
  });
});
