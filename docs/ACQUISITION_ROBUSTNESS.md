# Acquisition Robustness Evidence

This measured benchmark exercises the real frozen OpenCV/PatchCore agent through both sample-ID and uploaded-byte paths. Transforms are evaluation-only and never update the model.

**Status:** PASS  
**Generated:** 2026-10-22T19:26:31Z  
**Cases:** 15

| Sample | Truth | Acquisition variant | Decision | Path | Scale | Overlap |
|---|---:|---|---|---|---:|---:|
| `pcb1_borderline_01` | good | `sample_id_route` | **PASS** | `AGENT_CLEARED` | 1.0002 | 0.9750 |
| `pcb1_borderline_01` | good | `exact_image_bytes` | **PASS** | `AGENT_CLEARED` | 1.0002 | 0.9750 |
| `pcb1_borderline_01` | good | `png_reencode` | **PASS** | `AGENT_CLEARED` | 1.0002 | 0.9750 |
| `pcb1_borderline_01` | good | `browser_resize_0729` | **PASS** | `AGENT_CLEARED` | 0.7268 | 0.9698 |
| `pcb1_borderline_01` | good | `jpeg_q90` | **PASS** | `AGENT_CLEARED` | 0.9996 | 0.9743 |
| `pcb1_bad_missing_01` | bad | `sample_id_route` | **FAIL** | `AGENT_CONFIRMED` | 0.9948 | 0.9498 |
| `pcb1_bad_missing_01` | bad | `exact_image_bytes` | **FAIL** | `AGENT_CONFIRMED` | 0.9948 | 0.9498 |
| `pcb1_bad_missing_01` | bad | `png_reencode` | **FAIL** | `AGENT_CONFIRMED` | 0.9948 | 0.9498 |
| `pcb1_bad_missing_01` | bad | `browser_resize_0729` | **HUMAN** | `INCONCLUSIVE` | 0.7255 | 0.9494 |
| `pcb1_bad_missing_01` | bad | `jpeg_q90` | **HUMAN** | `INCONCLUSIVE` | 0.9910 | 0.9444 |
| `pcb1_bad_scratch_01` | bad | `sample_id_route` | **HUMAN** | `SUSPECTED_DEFECT` | 1.0126 | 0.9710 |
| `pcb1_bad_scratch_01` | bad | `exact_image_bytes` | **HUMAN** | `SUSPECTED_DEFECT` | 1.0126 | 0.9710 |
| `pcb1_bad_scratch_01` | bad | `png_reencode` | **HUMAN** | `SUSPECTED_DEFECT` | 1.0126 | 0.9710 |
| `pcb1_bad_scratch_01` | bad | `browser_resize_0729` | **HUMAN** | `INCONCLUSIVE` | 0.7349 | 0.9025 |
| `pcb1_bad_scratch_01` | bad | `jpeg_q90` | **HUMAN** | `SUSPECTED_DEFECT` | 1.0109 | 0.9695 |

## Safety assertions

- Exact uploaded bytes must match the built-in sample route in decision, path, and score.
- A known-good part may be passed or conservatively reviewed, but never automatically failed.
- A known-bad part may fail or be routed to a human, but never escape as PASS.
- Every run must remain inside the four-step agent budget.
- The reported 0.729× browser-resize regression must recover to PASS with explicit scale and overlap evidence.

