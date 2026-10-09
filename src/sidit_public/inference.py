"""Inference on one labelled SiD3M-format JSONL scan, without training machinery."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .dataset import MillerDataset, collate_fn_offset
from .model import SiDIT, SiDIT_CONFIGS
from .symmetry import SG_TO_CS, SG_TO_LAUE, SG_TO_PG

CS_NAMES = ['Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Trigonal', 'Hexagonal', 'Cubic']
PG_NAMES = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm', '-42m', '4/mmm',
            '3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm', '-6m2', '6/mmm',
            '23', 'm-3', '432', '-43m', 'm-3m']
LAUE_NAMES = ['-1', '2/m', 'mmm', '4/m', '4/mmm', '-3', '-3m', '6/m', '6/mmm', 'm-3', 'm-3m']


def load_config():
    return json.loads(Path(__file__).with_name('config.json').read_text())


def prepare_scan(path, config=None):
    """Use the archived loader with all augmentation disabled and deterministic voxel sampling."""
    config = config or load_config()
    path = Path(path).resolve()
    dataset = MillerDataset(
        paths=str(path.parent), miller_index_offset=-config['min_hkl'],
        augment_angle=False, augment_scale=False, abs_label=False,
        norm_scale=config['norm_scale'], fixed_clip_fraction=1.0,
        use_intensity_features=config['use_intensity_features'],
        intensity_log1p=config['intensity_log1p'], angle_normed=config['angle_normed'],
        grid_size=config['grid_size'], grid_sample_mode='val',
        lattice_mean=config['lattice_mean'], lattice_std=config['lattice_std'],
    )
    dataset.files = [str(path)]
    sample = next(iter(dataset))
    batch = collate_fn_offset([sample])
    if batch[0].numel() == 0:
        raise ValueError('The input has no usable diffraction points.')
    return batch, dataset


def load_model(weights, device='cuda', config=None):
    config = config or load_config()
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('SiDIT sparse/FlashAttention inference requires the configured CUDA environment.')
    model = SiDIT(in_channels=5, num_classes=config['max_hkl'] - config['min_hkl'] + 1,
                  cls_mode=False, use_intensity_features=True,
                  **SiDIT_CONFIGS[config['model_scale']]).to(device)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


def decode_cell(normalized, config=None):
    """Decode the archived length plus cosine/sine head in float32, as in the evaluator."""
    config = config or load_config()
    normalized = normalized.float()
    mean = normalized.new_tensor(config['lattice_mean'])
    std = normalized.new_tensor(config['lattice_std'])
    raw = normalized * std + mean
    cos, sin = raw[:, 3:6], raw[:, 6:9]
    norm = torch.sqrt(cos * cos + sin * sin).clamp_min(1e-8)
    cos, sin = cos / norm, sin / norm
    deg = torch.atan2(sin, cos) * 180.0 / np.pi
    return torch.cat([raw[:, :3], deg], dim=1)


@torch.inference_mode()
def infer_scan(weights, scan, device='cuda', seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = load_config()
    batch, _ = prepare_scan(scan, config)
    coords, feats, labels, offsets, crystal, info = batch
    model = load_model(weights, device, config)
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(coords.to(device), feats[:, :5].to(device), offsets.to(device),
                        intensity_stats=crystal['intensity_stats'].to(device))
    hkl = torch.stack([outputs[k].argmax(-1) + config['min_hkl'] for k in ['h', 'k', 'l']], dim=1)
    cs = int(outputs['cs_logits'].argmax(-1)[0])
    laue = int(outputs['laue_logits'].argmax(-1)[0])
    pg = int(outputs['pg_logits'].argmax(-1)[0])
    sg = int(outputs['space_group'].argmax(-1)[0]) + 1
    result = {'input_file': Path(scan).name, 'input_point_count': info[0]['grid_pre_count'],
              'evaluated_point_count': int(hkl.shape[0]), 'seed': seed,
              'source_point_indices': info[0]['grid_sample_index'].tolist(),
              'hkl_family': hkl.cpu().tolist(), 'crystal_system': CS_NAMES[SG_TO_CS[sg - 1]],
              'laue_class': LAUE_NAMES[SG_TO_LAUE[sg - 1]], 'point_group': PG_NAMES[SG_TO_PG[sg - 1]],
              'auxiliary_head_predictions': {'crystal_system': CS_NAMES[cs],
                                             'laue_class': LAUE_NAMES[laue], 'point_group': PG_NAMES[pg]},
              'space_group_number': sg,
              'space_group_top3': (outputs['space_group'].topk(3, dim=1).indices[0] + 1).cpu().tolist(),
              'cell_parameters_a_b_c_alpha_beta_gamma': decode_cell(outputs['lattice_params'], config)[0].cpu().tolist(),
              'scope': 'Canonical reflection-family labels and separate crystal predictions; no orientation-matrix refinement.'}
    reference_hkl = labels + config['min_hkl']
    result['reference_hkl_correct_points'] = int((hkl.cpu() == reference_hkl).all(dim=1).sum())
    result['reference_hkl_point_accuracy_percent'] = 100. * result['reference_hkl_correct_points'] / hkl.shape[0]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True, help='Tensor-only sidit-base-state-dict.pt')
    parser.add_argument('--input', required=True, help='One labelled SiD3M JSONL scan')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    result = infer_scan(args.weights, args.input, seed=args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['input_file', 'evaluated_point_count', 'space_group_number',
                                          'reference_hkl_point_accuracy_percent']}))


if __name__ == '__main__':
    main()
