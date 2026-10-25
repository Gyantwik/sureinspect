# Agentic Policy Decision Matrix

This artifact executes the frozen pure policy and proves that every advertised decision path is reachable under an explicit, reviewable input contract. It complements—not replaces—the live-image traces in `docs/AGENTIC_TRACE_EVIDENCE.md`.

OpenCV runtime: `5.0.0`

| Fixture | Decision | Path | Why |
|---|---|---|---|
| `quality_rejected` | **RETAKE** | `QUALITY_REJECTED` | Quality gate rejected frame before perception scoring |
| `fast_pass` | **PASS** | `FAST_PASS` | Perception score 1.8000 <= t_pass (2.0000) with normal illumination |
| `agent_cleared` | **PASS** | `AGENT_CLEARED` | Ambiguous perception score 2.1000 cleared: structural residual 0.2000 <= r_low (0.3000) confirms normal assembly |
| `agent_confirmed` | **FAIL** | `AGENT_CONFIRMED` | Perception score 2.5000 > t_fail (2.3000) and structural residual 0.6000 >= r_high (0.4800) confirm defect |
| `suspected_defect` | **HUMAN** | `SUSPECTED_DEFECT` | Structural residual 0.6000 >= r_high (0.4800) indicates defect, but score (2.1000) or lighting shift requires human review |
| `evidence_conflict` | **HUMAN** | `EVIDENCE_CONFLICT` | Perception score 2.5000 > t_fail (2.3000) indicates anomaly, but structural residual 0.2000 <= r_low (0.3000) shows normal variation |
| `lighting_shift_review` | **HUMAN** | `LIGHTING_SHIFT_REVIEW` | Perception score 1.8000 <= t_pass (2.0000), but factory illumination covariate shift requires review |
| `alignment_failed` | **HUMAN** | `ALIGN_FAILED` | Secondary verification required but template alignment failed (insufficient inliers) |
| `inconclusive` | **HUMAN** | `INCONCLUSIVE` | Structural residual 0.4000 lies in indeterminate zone (0.3000, 0.4800) |
| `budget_exhausted` | **HUMAN** | `BUDGET_EXHAUSTED` | Processing budget (4 steps or latency ceiling) exhausted before reaching a conclusive automated verdict |

Regenerate with:

```powershell
.venv\Scripts\python.exe -m eval.generate_agentic_evidence
```
