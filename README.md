# SiDIT

SiDIT jointly predicts canonical reflection-family labels, unit-cell parameters, and crystal symmetry from a single-crystal X-ray diffraction point cloud. This release provides the paper's SiDIT base model, one inference configuration and a compact interface for labelled SiD3M-format test scans.

The model predicts per-point `(h, k, l)` classes within `[-10, 10]`, a nine-component cell representation, and separate crystal-system, Laue-class, point-group, and space-group heads. It uses a Point Transformer V3 encoder–decoder, multiscale global pooling, and twelve intensity descriptors. The reflection-family labels follow the canonical labels stored in the supplied simulated data. Signed geometric indexing additionally requires a common orientation and diffraction refinement.

## Files and model

- `src/sidit_public/model.py`: SiDIT prediction model with portable package imports.
- `src/sidit_public/dataset.py`: JSONL loader and numerical preprocessing.
- `src/sidit_public/inference.py`: compact single-scan inference and output decoding.
- `src/sidit_public/vendor_pointcept/`: the required Pointcept point structure, module container, and serialization functions, with their MIT notice.
- `src/sidit_public/config.json`: inference configuration and cell normalization constants.
- [Model parameters on Hugging Face](https://huggingface.co/max6616/SiDIT/tree/v0.1.2): tensor-only checkpoint, approximately 195 MB.
- [Test support data on Hugging Face](https://huggingface.co/datasets/max6616/SiD3M-test/tree/jac-2026-10-12): released test scans and their provenance. Consult its dataset card for the exact recovered scope.

The checkpoint contains 523 model-state tensors. Optimizer state and Python argument objects are omitted. `SOURCE_IDENTITY.json` records the weight identity and implementation source hashes.

## Benchmark result support

[`results/`](results/) contains compact per-scan reflection counts, aligned cell/symmetry arrays, and the sensitivity-series aggregates. Reaggregation gives main-cohort reflection-family accuracy **88.10%**, SG top-1/top-3 **94.29%/99.81%**, and component-wise length/angle MAEs **0.19 Å/0.54°**. The OOD cohort gives **76.76%**, **61.21%/86.81%**, and **0.96 Å/0.65°**, respectively. The stored benchmark predictions and counts support reaggregation with explicit denominators. The result-support README explains the independent file orders and metric definitions.

[`results/INTERPRETATION.md`](results/INTERPRETATION.md) documents angularly ordered point thinning, sensitivity-series field definitions and the full-scan intensity descriptor used in those evaluations.

## Tested environment

The inference interface was tested in the Linux CUDA environment: Python `3.12.11`, PyTorch `2.8.0+cu128`, FlashAttention `2.8.2`, `torch-scatter 2.1.2+pt28cu128`, `spconv-cu124 2.3.8`, `timm 1.0.19`, and `addict 2.4.0`, on an NVIDIA RTX 5090. CUDA extension builds must match the selected PyTorch/CUDA installation. This package does not install or replace those GPU dependencies automatically.

After preparing that environment:

```bash
python -m pip install --no-deps .
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('max6616/SiDIT', 'sidit-base-state-dict.pt', revision='v0.1.2'))"
python -m sidit_public.inference \
  --weights /path/to/sidit-base-state-dict.pt \
  --input /path/to/test/scan.jsonl \
  --output prediction.json
```

The input JSONL record contains `input_sequence` rows `[rotation_angle_degrees, x, y, intensity]`, the corresponding canonical `labels`, and `metadata.crystal_params`. Coordinates `x` and `y` are normalized detector coordinates. The supplied test format includes reference metadata for the loader; those reference targets are used only for reporting agreement and are not passed to the model.

Output records preserve the selected original point indices, predicted reflection-family labels, three candidate space-group numbers, and `(a, b, c, alpha, beta, gamma)`. The reported hierarchy labels are mapped from the predicted space group, as in the benchmark evaluation; independent auxiliary-head predictions are included separately. Predicted lengths are in ångströms and angles in degrees.

## Numerical preprocessing

Evaluation uses no rotation or scale augmentation. Detector coordinates are rescaled about `(0.5, 0.5)` by `0.99 / max(abs(x, y))`. One point is retained per `1e-4` voxel of `(x, y, rotation_angle/360)`; the first point in each voxel is selected. The model features are `[sin(theta), cos(theta), x, y, log1p(intensity)/q99]`, with `theta` obtained by converting the input rotation angle from degrees to radians. The backbone uses an internal grid size `0.005`.

Intensity descriptors are computed from raw input intensities, before coordinate rescaling and grid sampling, unless the record provides the precomputed descriptors. Ten approximately equal-count shells are formed by sorting detector radius. Four circular correlations use detector azimuth, with 72 five-degree bins and shifts of 60, 90, 120, and 180 degrees. The nine-component cell head is denormalized in float32; cosine/sine pairs are normalized before `atan2` recovers the angles.

This release supports inference with the SiDIT base model and recovered test inputs. Training and validation data, experiment orchestration and manuscript plotting scripts are outside the release. The dataset card defines the recovered raw-input scope; benchmark prediction records are provided separately for statistical reaggregation. FP16 sparse GPU inference exhibited repeated-forward variation on a recovered scan, including an occasional adjacent-space-group change.

## Attribution

Authors: Zhao Zhang, Zheng Dong, Zhi Geng, Xinlong Dong, Yi Zhang, and Gaoqi He. Correspondence: Gaoqi He (`gqhe@cs.ecnu.edu.cn`). Implementation source: commit `208aed9`. Pointcept-derived functions retain the upstream copyright and MIT license.

See `CITATION.cff` for software citation metadata.
