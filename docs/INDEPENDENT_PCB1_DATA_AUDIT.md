# Independent PCB1 Data Search and Audit

## Outcome

No public, independently labelled defect set for the exact VisA PCB1 / HC-SR04 board design was found. The frozen split manifest already contains all 1,104 official VisA PCB1 images: 904 training normals, 100 test normals, and 100 anomalies. Re-downloading or re-splitting VisA would therefore not create independent evidence.

## Sources reviewed

- [Amazon VisA](https://registry.opendata.aws/visa/) is the source dataset already used by the project. Its PCB1 images are exhausted by the frozen manifest.
- [AmazonScience RobustAD](https://huggingface.co/datasets/AmazonScience/RobustAD) provides genuinely new, CC BY 4.0 PCB distribution-shift data, but its PCB is a VS1003/VS1053 MP3-decoder board rather than the HC-SR04 board. It is not PCB1 effectiveness evidence.
- [DeepPCB](https://github.com/tangsanli5201/DeepPCB) contains aligned bare-PCB crops and six etched defect classes. Its board layouts and task differ from assembled HC-SR04 inspection.
- [Wikimedia Commons ultrasonic range-finding category](https://commons.wikimedia.org/wiki/Category:Ultrasonic_range_finding) contains six rights-cleared HC-SR04 photographs suitable for a small external normal/acquisition development set.

## External normal development robustness audit

Four front-view blue HC-SR04 photographs form the primary cohort. A back-side view and a red product variant are reported separately. Acquisition records preserve source page, author metadata, license, source timestamp, dimensions, SHA-256, and Wikimedia SHA-1.

The evaluator checks every downloaded file against all 1,104 frozen VisA PCB1 SHA-256 and difference hashes. Exact overlap and near-duplicate overlap at Hamming distance 2 are both zero.

The current agent produced no automatic `FAIL`, but it also produced no `PASS`: the four primary images yielded one `RETAKE` and three `HUMAN`. Board-aware reframing allowed three images to pass acquisition quality checks; one remained correctly rejected for blur. None of the three admitted photographs produced trustworthy fixture-template alignment, so all were escalated. This is safe behavior with zero automated coverage and reveals the remaining product-revision/alignment boundary outside the calibrated fixture.

The agent now gives the same decision and detailed route for all six records across three consecutive runs (18 total inspections). Explicit alignment failure takes precedence over elapsed-time bookkeeping, eliminating the earlier `ALIGN_FAILED`/`BUDGET_EXHAUSTED` route variation.

## Claim boundary

The files are external to VisA at the image-source level, but they have now informed acquisition-normalization development and therefore are **not untouched independent final evidence**. The set is also normal-only, small, visibly labelled rather than electrically verified, and not blinded. It cannot estimate defect escape rate, recall, F1, or complete PCB1 task accuracy. A complete independent evaluation still requires newly captured normal and defective HC-SR04 boards, frozen before label access and evaluated once under the documented protocol.
