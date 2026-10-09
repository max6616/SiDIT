# Retained SiDIT evaluation records

These compact records support reaggregation of the paper's existing predictions.
They contain no training inputs, raw experimental measurements, optimizer state,
or new model-performance run. All scans are simulated.

Each `*-reflection-counts.jsonl.gz` contains one record per scan, ordered by
accuracy in the retained cache. SG numbers in `space_group` and `pred_sg` are
zero based: add one for the International Tables number. A filename may occur in
multiple source roots in the main cohort. The counts, rather than the filename,
identify each stored observation. Overall HKL is 100 times the sum of correct
triplets divided by the sum of eligible reflections. Group HKL uses the same
ratio within each true SG. Per-scan distributions use each count ratio once.

Each `*-cell-symmetry.npz` preserves the independent, original evaluation-array
order. Arrays in this file are aligned with each other. They are **not** aligned
row by row with the accuracy-sorted count file; do not join the two by row or by
the potentially repeated filename. Arrays contain denormalized cell lengths in
angstroms, cell angles in degrees, and zero-based true/predicted symmetry
classes. `y_pred_sg_top3` is a set of the three selected SG candidates; its column
order is not a ranking. CS, Laue and PG predictions were mapped from SG top-1.
Loading needs only NumPy with `allow_pickle=False`.

Length and angle MAEs average absolute errors over all scans and all three
components. ECDFs flatten the three component errors; their denominator is 3M.
SG top-1 counts equal true/predicted classes; SG top-3 checks membership in the
three selected candidates. Group cell MAEs use the aligned true SG arrays.

The main cohort contains 14,990 scans and 129,662,418 reflections: HKL 88.10%,
SG top-1/top-3 94.29%/99.81%, length/angle MAE 0.19 A/0.54 deg. The OOD cohort
contains 9,970 scans: HKL 76.76%, SG 61.21%/86.81%, MAE 0.96 A/0.65 deg.
Previously reported main HKL 90.47% and angle MAE 0.19 deg corresponded to
equal-weight space-group means. The paper now consistently uses the full-cohort
definitions above. Underlying stored predictions have not been changed.

`perturbations.json` contains retained aggregate sensitivity series. Each series
uses its own preprocessing pass, so its unperturbed reference is retained as
measured. No per-scan perturbation predictions are reconstructed from aggregates.
The original scan-input recovery scope is documented separately in the dataset
card; these cached predictions do not certify that every old raw input was found.
