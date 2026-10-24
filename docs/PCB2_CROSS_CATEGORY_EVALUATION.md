# PCB2 Cross-Category Evaluation

This is a zero-shot out-of-distribution stress test: the frozen PCB1 model, calibration, and PCB1 reference templates were applied to the official VisA PCB2 test split. PCB2 is not a supported category, so this result measures safe abstention—not PCB2 accuracy.

- Official registry: https://registry.opendata.aws/visa/
- Official one-class split: https://github.com/amazon-science/spot-diff/blob/main/split_csv/1cls.csv
- Dataset license: CC BY 4.0
- Images: 200 (100 normal, 100 anomaly)
- Pipeline fingerprint: `a241045168624da082502a1b36d78e6588902afad264f3cbe166ef6e3cbc8094`
- PCB2 acquisition manifest SHA-256: `d175c27011f96add7538a2486602141b985767d6b670fe9ab17cb88701116cca`

| Ground truth | PASS | FAIL | HUMAN | RETAKE |
|---|---:|---:|---:|---:|
| normal | 0 | 0 | 97 | 3 |
| anomaly | 0 | 0 | 100 | 0 |

## Safety interpretation

- Anomaly PASS escapes: **0/100**.
- Anomalies intercepted by FAIL/HUMAN/RETAKE: **100/100**.
- Normal automatic failures: **0/100**.
- Normal abstention rate: **100.0%**.

A high HUMAN rate is correct behavior here: PCB2 geometry is outside the PCB1 calibration and template set. Supporting PCB2 automation would require a separate PCB2 bank, calibration split, reference templates, quality thresholds, and a new independent evaluation.
