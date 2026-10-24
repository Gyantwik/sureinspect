# PCB1 External Normal-Image Development Robustness Audit

This is an externally sourced, normal-only development robustness audit using six rights-cleared Wikimedia Commons photographs of HC-SR04 modules. None is an exact or near-duplicate match for any image in the frozen VisA PCB1 split.

It is **not independent final evidence** because these images have informed robustness development. The boards are only visibly normal, electrical function is unverified, labels were not blinded, and there are no defective images.

- Candidate commit: `a882d717155af48897683b2c4d138d4828b3b683`
- Pipeline fingerprint: `a88c51f28e34ccb685566cebb5d78fdd53dcd9d256abed4ef7d307a4ea377f32`
- Acquisition manifest SHA-256: `c8d125d5e719dd5dfe5d11ff30b68aee09d0ef579b4b013ae5cc20e6f08b0c22`
- Primary eligible images (front-view blue HC-SR04): 4
- Supplemental challenge images (back view or red product variant): 2
- Source: https://commons.wikimedia.org/wiki/Category:Ultrasonic_range_finding

| PASS | FAIL | HUMAN | RETAKE |
|---:|---:|---:|---:|
| 0 | 0 | 3 | 1 |

- Normal automatic failure rate: **0.0%**
- Normal pass rate: **0.0%**
- Abstention rate: **100.0%**
- Automated coverage: **0.0%**
- Supplemental outcomes: `{'HUMAN': 1, 'RETAKE': 1}`
- Stable decision and route across three runs: **6/6**

## Interpretation

This audit measures behavior on externally photographed examples of the supported board family under uncontrolled backgrounds, viewpoints, lighting, and product variants. It can reveal unsafe normal-board rejection and acquisition brittleness, but it cannot measure defect interception.

The primary set is reported separately from the back-side view and red-board product variant so unsupported views do not distort the supported front-view result.

All records are replayed three times and exact decision/path stability is reported above.
