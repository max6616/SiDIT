# SiDIT

SiDIT jointly predicts canonical reflection-family labels, unit-cell parameters, and crystal symmetry from a single-crystal X-ray diffraction point cloud. This repository contains the historical model and a compact inference interface for labelled SiD3M-format test scans.

The model predicts per-point `(h, k, l)` classes within `[-10, 10]`, a nine-component cell representation, and separate crystal-system, Laue-class, point-group, and space-group heads. It uses a Point Transformer V3 encoder–decoder, multiscale global pooling, and twelve intensity descriptors. The reflection-family labels follow the canonical labels stored in the supplied simulated data. The interface does not produce an orientation matrix or perform diffraction refinement.

## Files and model

- `src/sidit_public/model.py`: archived SiDIT model, with package imports made portable.
- `src/sidit_public/dataset.py`: archived JSONL loader and numerical preprocessing.
- `src/sidit_public/inference.py`: compact single-scan inference and output decoding.
- `src/sidit_public/vendor_pointcept/`: the required Pointcept point structure, module container, and serialization functions, with their MIT notice.
- `src/sidit_public/config.json`: inference configuration and cell normalization constants.
- [Model parameters on Hugging Face](https://huggingface.co/max6616/SiDIT): tensor-only checkpoint, approximately 195 MB.
- [Test support data on Hugging Face](https://huggingface.co/datasets/max6616/SiD3M-test): released test scans and their provenance. Consult its dataset card for the exact recovered scope.

The released checkpoint preserves all 523 state tensors from the retained model associated with the cached benchmark. Optimizer state and Python argument objects are omitted. `SOURCE_IDENTITY.json` records tensor-file identity and the historical source hashes.

## Retained result support

[`results/`](results/) contains compact per-scan reflection counts, aligned cell/symmetry arrays, and the retained perturbation aggregates. Reaggregation gives main-cohort reflection-family accuracy **88.10%**, SG top-1/top-3 **94.29%/99.81%**, and component-wise length/angle MAEs **0.19 Å/0.54°**. The OOD cohort gives **76.76%**, **61.21%/86.81%**, and **0.96 Å/0.65°**, respectively. These are the existing cached predictions, reaggregated with explicit denominators. The result-support README explains the independent file orders and metric definitions.

## Tested environment

The exported core was checked in the retained Linux CUDA environment: Python `3.12.11`, PyTorch `2.8.0+cu128`, FlashAttention `2.8.2`, `torch-scatter 2.1.2+pt28cu128`, `spconv-cu124 2.3.8`, `timm 1.0.19`, and `addict 2.4.0`, on an NVIDIA RTX 5090. CUDA extension builds must match the selected PyTorch/CUDA installation. This package does not install or replace those GPU dependencies automatically.

After preparing that environment:

```bash
python -m pip install --no-deps .
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('max6616/SiDIT', 'sidit-base-state-dict.pt'))"
python -m sidit_public.inference \
  --weights /path/to/sidit-base-state-dict.pt \
  --input /path/to/test/scan.jsonl \
  --output prediction.json
```

The input JSONL record contains `input_sequence` rows `[rotation_angle_degrees, x, y, intensity]`, the corresponding canonical `labels`, and `metadata.crystal_params`. Coordinates `x` and `y` are normalized detector coordinates. The supplied test format includes reference metadata for the loader; those reference targets are used only for reporting agreement and are not passed to the model.

Output records preserve the selected original point indices, predicted reflection-family labels, three candidate space-group numbers, and `(a, b, c, alpha, beta, gamma)`. The reported hierarchy labels are mapped from the predicted space group, as in the retained evaluation; independent auxiliary-head predictions are included separately. Predicted lengths are in ångströms and angles in degrees.

## Numerical preprocessing

Evaluation uses no rotation or scale augmentation. Detector coordinates are rescaled about `(0.5, 0.5)` by `0.99 / max(abs(x, y))`. One point is retained per `1e-4` voxel of `(x, y, rotation_angle/360)`; the first point in each voxel is selected. The model features are `[sin(rotation_angle), cos(rotation_angle), x, y, log1p(intensity)/q99]`. The backbone uses its original internal grid size `0.005`.

Intensity descriptors are computed from raw input intensities, before coordinate rescaling and grid sampling, unless the record provides the archived precomputed descriptors. Ten approximately equal-count shells are formed by sorting detector radius. Four circular correlations use detector azimuth, with 72 five-degree bins and shifts of 60, 90, 120, and 180 degrees. The nine-component cell head is denormalized in float32; cosine/sine pairs are normalized before `atan2` recovers the angles.

This release supports inspection and inference with the retained model and recovered test inputs. Training and validation data, full experiment orchestration, and manuscript plotting scripts are outside this compact release. The dataset card states which original test inputs have been recovered; recovery of a subset does not establish full benchmark replay. The retained FP16 sparse GPU path exhibited repeated-forward variation on a recovered scan, including an occasional adjacent-space-group change; source/tensor identity and GPU execution were checked separately from numerical replay.

## Attribution

Authors: Zhao Zhang, Zheng Dong, Zhi Geng, Xinlong Dong, Yi Zhang, and Gaoqi He. Correspondence: Gaoqi He (`gqhe@cs.ecnu.edu.cn`). The original SiDIT implementation is preserved from historical XRDT source commit `208aed9`. Pointcept-derived functions retain the upstream copyright and MIT license.

See `CITATION.cff` for software citation metadata.
