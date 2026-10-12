# Sensitivity-series interpretation

The `sampling_interval_deg` field in `perturbations.json` names the
point-thinning factor 1, 2, 4 or 8. The evaluator sorts reflections by rotation
angle and retains every kth point. This operation reduces spot density; it does
not select frames at k-degree intervals.

Coverage, point-thinning and corruption series modify the point encoder input.
The independent intensity descriptor supplied directly to symmetry classification
is retained from the complete original scan. HKL and cell predictions depend on
the modified point stream. These series therefore describe point-stream sensitivity
with the full-scan intensity descriptor available to symmetry prediction.

Corruption level q=0,...,5 sets detector-coordinate jitter and point-replacement
probability ceilings to 0.002q and 0.05q. Strengths interpolate from zero to those
ceilings within each mini-batch. Randomly replaced points are excluded from the
HKL denominator. The training replacement ceiling was 0.2; the level-5 evaluation
ceiling is 0.25. The series use these specified point-stream conditions.
