import argparse
import json
import os
import random
import glob
import math
import concurrent.futures
import warnings
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import IterableDataset, DataLoader

INTENSITY_FEATURE_KEYS = [
    'wilson', 'e2_std', 'cdf_1', 'cdf_2', 'cdf_3',
    'gini', 'skewness', 'kurtosis',
    'c_60', 'c_90', 'c_120', 'c_180',
]

def compute_intensity_statistics(points_tensor, num_shells=10):
    """
    Compute comprehensive intensity statistics for a diffraction pattern.
    
    Args:
        points_tensor: [N, C] where structure is [sin(ang), cos(ang), x, y, intensity_raw]
        num_shells: number of resolution shells for E^2 calculation
    
    Returns:
        e_squared: [N] tensor of normalized structure factors E^2
        global_stats: dict containing Wilson stats, CDF values, Gini, skewness, kurtosis, angular correlations
    """
    # Extract coordinates and intensity
    # Structure: [sin(ang), cos(ang), x, y, intensity_raw]
    x = points_tensor[:, 2]
    y = points_tensor[:, 3]
    I = points_tensor[:, 4]  # raw intensity
    
    # Calculate radius from center
    dx = x - 0.5
    dy = y - 0.5
    r = torch.sqrt(dx**2 + dy**2) / 0.70710678118
    
    # Calculate phi for angular correlation
    phi = torch.atan2(dy, dx)
    
    # Ensure intensity is positive
    I = torch.clamp(I, min=1e-8)
    
    # Shell-based E^2 calculation
    # Divide r into shells
    if r.numel() < num_shells:
        num_shells = max(1, r.numel() // 2)
    
    r_sorted, sort_idx = torch.sort(r)
    I_sorted = I[sort_idx]
    
    points_per_shell = len(r) // num_shells
    e_squared = torch.zeros_like(I)
    
    for i in range(num_shells):
        start = i * points_per_shell
        end = (i + 1) * points_per_shell if i < num_shells - 1 else len(r)
        shell_I = I_sorted[start:end]
        mean_I = shell_I.mean()
        if mean_I > 1e-8:
            shell_E2 = shell_I / mean_I
        else:
            shell_E2 = torch.ones_like(shell_I)
        e_squared[sort_idx[start:end]] = shell_E2
    
    # Wilson statistics
    wilson_stat = torch.abs(e_squared - 1.0).mean()
    e2_std = e_squared.std()
    
    # CDF calculation
    cdf_1 = (e_squared <= 1.0).float().mean()
    cdf_2 = (e_squared <= 2.0).float().mean()
    cdf_3 = (e_squared <= 3.0).float().mean()
    
    # Gini coefficient
    e2_sorted = torch.sort(e_squared)[0]
    n = len(e2_sorted)
    index = torch.arange(1, n + 1, dtype=torch.float32, device=e_squared.device)
    gini = (2 * torch.sum(index * e2_sorted)) / (n * torch.sum(e2_sorted)) - (n + 1) / n
    
    # Skewness and Kurtosis
    e2_mean = e_squared.mean()
    e2_std_val = e_squared.std() + 1e-8
    skewness = ((e_squared - e2_mean) ** 3).mean() / (e2_std_val ** 3)
    kurtosis = ((e_squared - e2_mean) ** 4).mean() / (e2_std_val ** 4) - 3.0
    
    # Angular correlation
    # Bin by phi and compute autocorrelation
    num_bins = 72  # 5 degree bins
    phi_normalized = (phi + math.pi) / (2 * math.pi)  # normalize to [0, 1]
    phi_bins = (phi_normalized * num_bins).long()
    phi_bins = torch.clamp(phi_bins, 0, num_bins - 1)
    
    # Average intensity per bin
    bin_intensities = torch.zeros(num_bins, device=I.device)
    bin_counts = torch.zeros(num_bins, device=I.device)
    for i in range(len(I)):
        bin_idx = phi_bins[i]
        bin_intensities[bin_idx] += I[i]
        bin_counts[bin_idx] += 1
    
    bin_counts = torch.clamp(bin_counts, min=1.0)
    bin_avg = bin_intensities / bin_counts
    
    # Compute correlations at specific angles
    def circular_corr(signal, shift):
        rolled = torch.roll(signal, shift)
        return torch.corrcoef(torch.stack([signal, rolled]))[0, 1].item()
    
    c_60 = circular_corr(bin_avg, num_bins // 6)   # 60 degrees
    c_90 = circular_corr(bin_avg, num_bins // 4)   # 90 degrees
    c_120 = circular_corr(bin_avg, num_bins // 3)  # 120 degrees
    c_180 = circular_corr(bin_avg, num_bins // 2)  # 180 degrees
    
    global_stats = {
        'wilson': wilson_stat.item(),
        'e2_std': e2_std.item(),
        'cdf_1': cdf_1.item(),
        'cdf_2': cdf_2.item(),
        'cdf_3': cdf_3.item(),
        'gini': gini.item(),
        'skewness': skewness.item(),
        'kurtosis': kurtosis.item(),
        'c_60': c_60,
        'c_90': c_90,
        'c_120': c_120,
        'c_180': c_180,
    }
    
    return e_squared, global_stats

def _normalize_intensity_stats_payload(stats_payload):
    """
    Normalize precomputed intensity stats payload to a dict with fixed keys.
    Accepts:
      - dict with INTENSITY_FEATURE_KEYS
      - list/tuple with length 12 in INTENSITY_FEATURE_KEYS order
    Returns:
      - normalized dict or None
    """
    if stats_payload is None:
        return None

    if isinstance(stats_payload, (list, tuple)):
        if len(stats_payload) != len(INTENSITY_FEATURE_KEYS):
            return None
        out = {}
        for k, v in zip(INTENSITY_FEATURE_KEYS, stats_payload):
            v = float(v)
            if not math.isfinite(v):
                v = 0.0
            out[k] = v
        return out

    if isinstance(stats_payload, dict):
        out = {}
        for k in INTENSITY_FEATURE_KEYS:
            if k not in stats_payload:
                return None
            v = float(stats_payload[k])
            if not math.isfinite(v):
                v = 0.0
            out[k] = v
        return out

    return None

def compute_intensity_stats_from_input_sequence(input_sequence, angle_normed=False):
    """
    Compute intensity statistics from raw input_sequence records.
    input_sequence expected shape: [N, 4+] with [angle, x, y, intensity_raw].
    """
    points_tensor = torch.tensor(input_sequence, dtype=torch.float)
    if points_tensor.ndim == 3 and points_tensor.shape[1] == 1:
        points_tensor = points_tensor[:, 0]
    if points_tensor.ndim != 2 or points_tensor.shape[1] < 4 or points_tensor.shape[0] == 0:
        return None

    angle_deg = points_tensor[:, 0]
    if bool(angle_normed):
        angle_deg = angle_deg * 360.0
    angle_rad = angle_deg * (math.pi / 180.0)
    sin_angle = torch.sin(angle_rad)
    cos_angle = torch.cos(angle_rad)
    x = points_tensor[:, 1]
    y = points_tensor[:, 2]
    intensity_raw = torch.clamp(points_tensor[:, 3], min=0.0)
    points_raw = torch.stack([sin_angle, cos_angle, x, y, intensity_raw], dim=1)

    _, stats = compute_intensity_statistics(points_raw, num_shells=10)
    return _normalize_intensity_stats_payload(stats)

class MillerDataset(IterableDataset):
    def __init__(self, paths, miller_index_offset, augment_angle=False, augment_scale=False, 
                 scale_range=(0.9, 1.1), max_points=65535, debug=0, abs_label=False, norm_scale=False, fixed_clip_fraction=None, fixed_density_divisor=None,
                 lattice_stats_json=None, lattice_mean=None, lattice_std=None,
                 cls_mode=False, use_intensity_features=False, intensity_log1p=True, angle_normed=False,
                 grid_size=None, grid_sample_mode="train", seed=42):
        super(MillerDataset, self).__init__()
        self._base_seed = seed
        
        self.cls_mode = cls_mode
        self.use_intensity_features = use_intensity_features
        self.intensity_log1p = bool(intensity_log1p)
        self.angle_normed = bool(angle_normed)
        self._intensity_cache = {}
        
        # Support single path or list of paths
        if isinstance(paths, str):
            self.paths = [paths]
        else:
            self.paths = paths
            
        self.files = []

        for path in self.paths:
            if not os.path.isdir(path):
                print(f"Warning: Not a valid directory, skipped: {path}")
                continue
            path_files = glob.glob(os.path.join(path, '**', '*.jsonl'), recursive=True)
            if not path_files:
                print(f"Warning: No '.jsonl' files found in directory {path}.")
            else:
                self.files.extend(path_files)
        
        if debug > 0: 
            self.files = self.files[:debug]
            print(f"--> Debug mode: limited to first {debug} files")
            
        if not self.files:
            raise ValueError(f"No '.jsonl' files found in any provided path: {self.paths}")

        # Sort files by size for DDP load balancing: interleaved rank sharding
        # on a size-sorted list ensures each rank gets an even mix of large/small samples,
        # reducing idle time from variable-length point clouds.
        try:
            self.files.sort(key=lambda f: os.path.getsize(f))
        except OSError:
            pass

        self.augment_angle = augment_angle
        self.augment_scale = augment_scale
        self.scale_range = scale_range
        self.max_points = max_points
        self.miller_index_offset = miller_index_offset
        self.abs_label = abs_label
        self.norm_scale = norm_scale
        self.fixed_clip_fraction = fixed_clip_fraction
        self.fixed_density_divisor = fixed_density_divisor
        if grid_size is not None:
            if grid_size <= 0:
                grid_size = None
            else:
                assert grid_size > 0, "grid_size must be positive"
        assert grid_sample_mode in ["train", "val"], "grid_sample_mode must be 'train' or 'val'"
        self.grid_size = grid_size
        self.grid_sample_mode = grid_sample_mode

        # Lattice normalization stats (mean/std per a,b,c,alpha,beta,gamma)
        self.lattice_mean = None
        self.lattice_std = None
        if lattice_mean is not None and lattice_std is not None:
            self.lattice_mean = torch.tensor(list(lattice_mean), dtype=torch.float)
            self.lattice_std = torch.tensor(list(lattice_std), dtype=torch.float)
        elif lattice_stats_json is not None and os.path.isfile(lattice_stats_json):
            try:
                with open(lattice_stats_json, 'r', encoding='utf-8') as f:
                    stats_payload = json.load(f)
                stats = stats_payload.get('stats', {})
                order = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
                means = [float(stats[k].get('mean', 0.0)) for k in order]
                stds = [float(stats[k].get('std', 1.0)) for k in order]
                
                # Update: Use Cosine for angles.
                # Force mean=0, std=1 for angles so they remain in [-1, 1] (roughly) without shifting
                # Indices 3, 4, 5 correspond to alpha, beta, gamma
                for idx in [3, 4, 5]:
                    means[idx] = 0.0
                    stds[idx] = 1.0

                # Avoid zero std
                stds = [s if abs(s) > 1e-8 else 1.0 for s in stds]
                self.lattice_mean = torch.tensor(means, dtype=torch.float)
                self.lattice_std = torch.tensor(stds, dtype=torch.float)
                print(f"--> Loaded lattice stats from {lattice_stats_json} (Angles forced to Mean=0, Std=1 for Cosine)")
            except Exception as e:
                print(f"Warning: failed to load lattice stats '{lattice_stats_json}': {e}")
                self.lattice_mean, self.lattice_std = None, None
        # Fallback: legacy coarse scaling if no stats provided
        if self.lattice_mean is None or self.lattice_std is None:
            self.lattice_mean = torch.zeros(9, dtype=torch.float)
            # Default for Cosine/Sine angles: std=1.0
            self.lattice_std = torch.tensor([10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float)
            print("--> Lattice stats not provided; using coarse scaling [0 mean, 9-dim]")
        
        print(f"--> Total files loaded: {len(self.files)}")
        if self.norm_scale:
            print("--> Coordinate normalization enabled")
        else:
            print("--> Coordinate normalization disabled")

        # Load precomputed intensity features cache (if available)
        if self.use_intensity_features:
            for path in self.paths:
                cache_path = os.path.join(path, 'intensity_features_cache.json')
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, 'r') as f:
                            cache_data = json.load(f)
                        for rel_path, features in cache_data.items():
                            abs_path = os.path.normpath(os.path.join(path, rel_path))
                            self._intensity_cache[abs_path] = features
                        print(f"--> Loaded intensity features cache: {cache_path} ({len(cache_data)} entries)")
                    except Exception as e:
                        print(f"--> Warning: failed to load intensity cache {cache_path}: {e}")
            if self._intensity_cache:
                print(f"--> Total cached intensity features: {len(self._intensity_cache)}")
            else:
                print("--> No intensity features cache found, will compute on-the-fly")

    def set_distributed(self, rank, world_size):
        """Configure DDP rank-based file sharding."""
        self._ddp_rank = rank
        self._ddp_world_size = world_size

    def set_epoch(self, epoch):
        """Set epoch for deterministic per-epoch shuffling (DDP-safe)."""
        self._epoch = epoch

    def __len__(self):
        total = len(self.files)
        world_size = getattr(self, '_ddp_world_size', 1)
        if world_size > 1:
            # All ranks must report the same length to avoid DDP iteration desync.
            # Use ceil division so every rank iterates the same number of files.
            return -(-total // world_size)  # ceil(total / world_size)
        return total

    def _apply_augmentations(self, points, labels):
        assert isinstance(points, (list, tuple)), "points must be a list/tuple of [angle, x, y, intensity]"
        assert len(points) > 0, "points must be non-empty"
        points_tensor_raw = torch.tensor(points, dtype=torch.float)
        labels_tensor_raw = torch.tensor(labels, dtype=torch.long)
        if points_tensor_raw.ndim == 3 and points_tensor_raw.shape[1] == 1:
            points_tensor_raw = points_tensor_raw[:, 0]
        if labels_tensor_raw.ndim == 3 and labels_tensor_raw.shape[1] == 1:
            labels_tensor_raw = labels_tensor_raw[:, 0]
        assert points_tensor_raw.ndim == 2 and points_tensor_raw.shape[1] >= 4, "points must be [N, 4+] with angle/x/y/intensity"
        assert torch.isfinite(points_tensor_raw[:, 0]).all(), "angle must be finite"
        if self.angle_normed:
            assert (points_tensor_raw[:, 0] >= 0.0).all() and (points_tensor_raw[:, 0] <= 1.0).all(), \
                "angle must be in [0, 1] when angle_normed=True"
            points_tensor_raw[:, 0] = points_tensor_raw[:, 0] * 360.0
        else:
            # Ensure angles are likely in degrees [0, 360) and NOT normalized to [0, 1]
            assert (points_tensor_raw[:, 0] >= 0.0).all() and (points_tensor_raw[:, 0] < 360.0).all(), \
                "angle must be in [0, 360) degrees"
            if points_tensor_raw.shape[0] > 100: # only check if enough points to be statistically significant
                 assert (points_tensor_raw[:, 0] > 1.0).any(), \
                    "angle seems normalized (all <= 1.0), but expected raw degrees [0, 360)"
        assert labels_tensor_raw.ndim in (2, 3), "labels must be [N,3+] or [H,N,3+]"
        if labels_tensor_raw.ndim == 2:
            assert labels_tensor_raw.shape[1] >= 3, "labels must be [N, 3+] with h/k/l"
            assert points_tensor_raw.shape[0] == labels_tensor_raw.shape[0], "points and labels must have same length"
        else:
            assert labels_tensor_raw.shape[2] >= 3, "labels must be [H, N, 3+] with h/k/l"
            assert points_tensor_raw.shape[0] == labels_tensor_raw.shape[1], "points and labels must have same length"
        points_tensor = points_tensor_raw.clone()
        labels_tensor = labels_tensor_raw.clone()
        orig_index = torch.arange(points_tensor.shape[0], dtype=torch.long)
        
        def _select_labels(label_tensor, index_or_mask):
            if label_tensor.ndim == 2:
                return label_tensor[index_or_mask]
            return label_tensor[:, index_or_mask]

        if self.norm_scale:
            # Expect x,y roughly within [0, 1] when normalization is enabled
            assert torch.isfinite(points_tensor[:, 1:3]).all(), "x/y must be finite"
            assert (points_tensor[:, 1:3] >= 0.0).all() and (points_tensor[:, 1:3] <= 1.0).all(), \
                "x/y must be in [0, 1] when norm_scale=True"
            max_xy = torch.max(torch.abs(points_tensor[:, 1:3]))
            if max_xy > 0:
                scale_factor = 0.99 / max_xy
                points_tensor[:, 1:3] = (points_tensor[:, 1:3] - 0.5) * scale_factor + 0.5

        if self.augment_angle:
            # angle is in degrees [0, 360); augment by cyclic shift in degrees
            assert torch.isfinite(points_tensor[:, 0]).all(), "angle must be finite"
            assert (points_tensor[:, 0] >= 0.0).all() and (points_tensor[:, 0] < 360.0).all(), \
                "angle must be in [0, 360) degrees"
            shift_deg = random.random() * 360.0
            points_tensor[:, 0] = (points_tensor[:, 0] + shift_deg) % 360.0
        
        # Deterministic angle clipping (for eval) has higher priority than random clipping
        if self.fixed_clip_fraction is not None:
            clip_fraction = float(self.fixed_clip_fraction)
            assert 0.0 < clip_fraction <= 1.0, "fixed_clip_fraction must be in (0, 1]"
            angle_frac = (points_tensor[:, 0] % 360.0) / 360.0
            mask = angle_frac <= clip_fraction
            assert mask.sum() >= 0
            points_tensor = points_tensor[mask]
            labels_tensor = _select_labels(labels_tensor, mask)
            orig_index = orig_index[mask]

        if self.augment_scale:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            points_tensor[:, 1:3] *= scale
            mask = (points_tensor[:, 1] >= 0) & (points_tensor[:, 1] <= 1) & (points_tensor[:, 2] >= 0) & (points_tensor[:, 2] <= 1)
            points_tensor = points_tensor[mask]
            labels_tensor = _select_labels(labels_tensor, mask)
            orig_index = orig_index[mask]
            if points_tensor.shape[0] == 0:
                return points_tensor_raw, orig_index, labels_tensor_raw, points_tensor_raw.shape[0], points_tensor_raw.shape[0]
        
        # Deterministic uniform angular downsampling by divisor (keep every k-th by sorted angle)
        if self.fixed_density_divisor is not None and int(self.fixed_density_divisor) > 1:
            k = int(self.fixed_density_divisor)
            order = torch.argsort(points_tensor[:, 0])
            kept = order[::k]
            if kept.numel() == 0:
                kept = order[:1]
            points_tensor = points_tensor[kept]
            labels_tensor = _select_labels(labels_tensor, kept)
            orig_index = orig_index[kept]

        pre_grid_count = points_tensor.shape[0]
        if self.grid_size is not None:
            # GridSample equivalent to Pointcept: one point per voxel
            angle_frac = (points_tensor[:, 0] % 360.0) / 360.0
            coord = torch.stack([points_tensor[:, 1], points_tensor[:, 2], angle_frac], dim=1)
            voxel = torch.floor(coord / float(self.grid_size)).to(torch.long)
            voxel = voxel - voxel.min(0).values
            _, inverse = torch.unique(voxel, dim=0, return_inverse=True)
            num_vox = int(inverse.max().item()) + 1 if inverse.numel() > 0 else 0
            if num_vox > 0:
                counts = torch.bincount(inverse, minlength=num_vox)
                idx_sorted = torch.argsort(inverse)
                cumsum = torch.cumsum(counts, 0)
                starts = torch.cat([torch.zeros(1, dtype=cumsum.dtype), cumsum[:-1]])
                valid = counts > 0
                if self.grid_sample_mode == "train":
                    rand_offsets = (torch.rand(num_vox) * counts.float()).long()
                    rand_offsets = torch.clamp(rand_offsets, max=counts - 1)
                    selected = idx_sorted[starts[valid] + rand_offsets[valid]]
                else:
                    selected = idx_sorted[starts[valid]]
                if selected.numel() > 0:
                    points_tensor = points_tensor[selected]
                    labels_tensor = _select_labels(labels_tensor, selected)
                    orig_index = orig_index[selected]

        post_grid_count = points_tensor.shape[0]
        return points_tensor, orig_index, labels_tensor, pre_grid_count, post_grid_count
        
    def __iter__(self):
        all_files = list(self.files)

        # DDP rank sharding: each rank gets a disjoint subset of files
        ddp_rank = getattr(self, '_ddp_rank', 0)
        ddp_world_size = getattr(self, '_ddp_world_size', 1)
        if ddp_world_size > 1:
            rank_files = all_files[ddp_rank::ddp_world_size]
            # Pad so every rank has the same number of files (avoids DDP iteration desync)
            max_per_rank = -(-len(all_files) // ddp_world_size)  # ceil division
            if len(rank_files) < max_per_rank and len(rank_files) > 0:
                # Wrap around to fill the gap (at most world_size-1 extra files)
                rank_files = rank_files + rank_files[:max_per_rank - len(rank_files)]
            all_files = rank_files

        # Worker sharding (within each rank)
        worker_info = torch.utils.data.get_worker_info()
        files_to_process = all_files if worker_info is None else all_files[worker_info.id::worker_info.num_workers]

        # Deterministic per-epoch shuffle (same seed across workers for consistency)
        epoch = getattr(self, '_epoch', 0)
        worker_id = worker_info.id if worker_info is not None else 0
        rng = random.Random(self._base_seed + epoch * 1000 + ddp_rank * 100 + worker_id)
        files_to_process = list(files_to_process)
        rng.shuffle(files_to_process)
        
        for f_path in files_to_process:
            filename = os.path.basename(f_path)
            try:
                with open(f_path, 'r') as f:
                    line = f.readline()
                    data = json.loads(line)
                    assert 'input_sequence' in data and 'labels' in data and 'metadata' in data, "missing required keys"
                    assert len(data['input_sequence']) == len(data['labels']), f"input_sequence: {len(data['input_sequence'])}, labels: {len(data['labels'])}"
                    # INSERT_YOUR_CODE
                    if 'metadata' in data and 'crystal_params' in data['metadata']:
                        data['metadata']['crystal_params'].pop('_symmetry_space_group_name_H-M', None)
                    assert 'crystal_params' in data['metadata'], "metadata.crystal_params is required"
                    assert len(data['metadata']['crystal_params'].values()) == 7, f"metadata: {data['metadata']}"

                    points, sampled_index, labels_raw, pre_grid_count, post_grid_count = self._apply_augmentations(
                        data['input_sequence'], data['labels']
                    )
                    if labels_raw.ndim == 2:
                        labels = labels_raw
                    else:
                        labels = labels_raw[0]
                    assert sampled_index.numel() == points.shape[0], "sampled_index and points must align"
                    assert labels.shape[0] == points.shape[0], f"labels.shape: {labels.shape}, points.shape: {points.shape}"
                    assert labels.ndim == 2 and labels.shape[1] >= 3, "labels must be [N, 3+] after preprocessing"
                
                if self.cls_mode:
                    # In classification mode, we don't need point-wise labels. 
                    # Use dummy labels to save memory/processing if possible, 
                    # but keep shape [N, 3] to satisfy collate_fn and model interface.
                    # We can just leave them as is or zero them out. 
                    # Zeroing them out avoids confusion.
                    labels = torch.zeros_like(labels)

                num_points = points.shape[0]
                if num_points == 0:
                    # Yield empty sample to keep iteration count consistent across DDP ranks
                    yield (torch.zeros(0, 6), torch.zeros(0, 3, dtype=torch.long),
                           {'lattice': torch.zeros(9), 'space_group': torch.tensor([0], dtype=torch.long)},
                           {'filename': filename, 'source_path': f_path, 'intensity_stats': None,
                            'grid_sample_index': torch.tensor([], dtype=torch.long),
                            'grid_pre_count': 0, 'grid_post_count': 0})
                    continue
                
                # Input format: [angle_deg, x, y, intensity_raw]
                # Convert to: [sin(angle), cos(angle), x, y, intensity_raw]
                angle_deg = points[:, 0]
                assert torch.isfinite(angle_deg).all(), "angle must be finite"
                assert (angle_deg >= 0.0).all() and (angle_deg < 360.0).all(), \
                    f"angle must be in [0, 360) degrees for {filename}; min={float(angle_deg.min())}, max={float(angle_deg.max())}"
                # If not clipping to a tiny fraction, we expect to see values > 1.0 in degree mode
                if (not self.angle_normed) and (self.fixed_clip_fraction is None or self.fixed_clip_fraction > 0.01) and angle_deg.numel() > 100:
                    assert (angle_deg > 1.0).any(), f"angle seems normalized (all <= 1.0) for {filename}, but expected degrees"
                angle_rad = angle_deg * (math.pi / 180.0)
                sin_angle = torch.sin(angle_rad)
                cos_angle = torch.cos(angle_rad)
                x = points[:, 1]
                y = points[:, 2]
                assert torch.isfinite(x).all() and torch.isfinite(y).all(), "x/y must be finite"
                intensity_raw = torch.clamp(points[:, 3], min=0.0)
                
                if labels.shape[1] == 6:
                    labels = labels[:, :3]

                crystal_params_raw = data.get('metadata', {}).get('crystal_params', None)
                assert crystal_params_raw is not None, "crystal_params missing in metadata"
                raw_vals_6 = torch.tensor([float(p) for p in crystal_params_raw.values()][:6], dtype=torch.float)
                assert raw_vals_6.numel() == 6, "crystal_params must provide 6 cell params"
                
                # Construct 9-dim vector: [a, b, c, cos_alpha, cos_beta, cos_gamma, sin_alpha, sin_beta, sin_gamma]
                raw_vals = torch.zeros(9, dtype=torch.float)
                raw_vals[:3] = raw_vals_6[:3] # a, b, c
                
                angles_rad = raw_vals_6[3:6] * (math.pi / 180.0)
                raw_vals[3:6] = torch.cos(angles_rad)
                raw_vals[6:9] = torch.sin(angles_rad)

                # Standardize using dataset stats
                if self.lattice_mean.shape[0] == 6:
                    lm = torch.cat([self.lattice_mean, torch.zeros(3)])
                    ls = torch.cat([self.lattice_std, torch.ones(3)])
                else:
                    lm, ls = self.lattice_mean, self.lattice_std

                lattice_labels = (raw_vals - lm) / ls
                assert '_symmetry_Int_Tables_number' in crystal_params_raw, "missing space group number"
                sg_num = int(crystal_params_raw['_symmetry_Int_Tables_number'])
                sg_label = torch.tensor([sg_num - 1], dtype=torch.long)
                crystal_labels = {'lattice': lattice_labels, 'space_group': sg_label}

                assert torch.max(sg_label) <= 229 and torch.min(sg_label) >= 0

                # Intensity feature engineering (if enabled)
                # Use raw intensity for physical statistics
                intensity_stats = None
                if self.use_intensity_features:
                    # Priority 1: per-sample precomputed field in jsonl
                    intensity_stats = _normalize_intensity_stats_payload(data.get('intensity_stats', None))

                    # Priority 2: legacy metadata field (for compatibility)
                    if intensity_stats is None:
                        intensity_stats = _normalize_intensity_stats_payload(
                            data.get('metadata', {}).get('intensity_stats', None)
                        )

                    # Priority 3: old directory-level cache file
                    if intensity_stats is None:
                        _cache_key = os.path.normpath(f_path)
                        _cached = self._intensity_cache.get(_cache_key, None)
                        intensity_stats = _normalize_intensity_stats_payload(_cached)

                    # Fallback: on-the-fly computation from raw input_sequence
                    # (same source and function as precompute script)
                    if intensity_stats is None:
                        intensity_stats = compute_intensity_stats_from_input_sequence(
                            data['input_sequence'],
                            angle_normed=self.angle_normed
                        )
                
                # Point-wise intensity preprocessing:
                # optional log1p compression + 99th percentile normalization
                if self.intensity_log1p:
                    intensity_proc = torch.log1p(intensity_raw)
                else:
                    intensity_proc = intensity_raw
                p99 = torch.quantile(intensity_proc, 0.99)
                if torch.isfinite(p99) and p99 > 0:
                    intensity_proc = intensity_proc / p99
                
                # Final format for model input: [sin(ang), cos(ang), x, y, intensity_proc, angle_deg]
                # angle_deg is kept at the end for data augmentation helpers in train.py
                points = torch.stack([sin_angle, cos_angle, x, y, intensity_proc, angle_deg], dim=1)

                if points.shape[0] > self.max_points:
                    points = points[:self.max_points]
                    labels = labels[:self.max_points, :]
                
                if self.abs_label:
                    labels = torch.abs(labels)
                else:
                    labels += self.miller_index_offset
                
                # Sample info
                sample_info = {
                    'filename': filename,
                    'source_path': f_path,
                    'intensity_stats': intensity_stats,  # Add intensity statistics
                    'grid_sample_index': sampled_index,
                    'grid_pre_count': int(pre_grid_count),
                    'grid_post_count': int(post_grid_count),
                }
                if self.grid_size is not None and pre_grid_count > 0:
                    ratio = float(post_grid_count) / float(pre_grid_count)
                    if ratio < 0.9:
                        warnings.warn(
                            f"GridSampling kept {post_grid_count}/{pre_grid_count} points (<90%) for {filename}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                yield points, labels, crystal_labels, sample_info
            except (json.JSONDecodeError, KeyError, AssertionError, ValueError, OSError) as e:
                import logging
                logging.warning(f"Skipping corrupted/invalid file {f_path}: {type(e).__name__}: {e}")
                # Yield empty sample to keep iteration count consistent across DDP ranks
                yield (torch.zeros(0, 6), torch.zeros(0, 3, dtype=torch.long),
                       {'lattice': torch.zeros(9), 'space_group': torch.tensor([0], dtype=torch.long)},
                       {'filename': filename, 'source_path': f_path, 'intensity_stats': None,
                        'grid_sample_index': torch.tensor([], dtype=torch.long),
                        'grid_pre_count': 0, 'grid_post_count': 0})


def collate_fn_offset(batch):
    points_list, labels_list, crystal_labels_list, sample_info_list = zip(*batch)
    
    valid_indices = [i for i, p in enumerate(points_list) if p.shape[0] > 0]
    if not valid_indices:
        empty_crystal_labels = {'lattice': torch.tensor([]), 'space_group': torch.tensor([], dtype=torch.long)}
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([], dtype=torch.int), empty_crystal_labels, []

    points_list = [points_list[i] for i in valid_indices]
    labels_list = [labels_list[i] for i in valid_indices]
    crystal_labels_list = [crystal_labels_list[i] for i in valid_indices]
    sample_info_list = [sample_info_list[i] for i in valid_indices]

    # points format: [sin(ang), cos(ang), x, y, intensity_proc, angle_deg]
    # user requested:
    # Feature input: [sin(ang), cos(ang), x, y, intensity_proc] (5-dim)
    # Coordinate input: [angle_deg/360.0, x, y] (3-dim, normalized to 0-1)
    # We keep angle_deg at the end of feats for data augmentation.
    
    # coords: [x, y, angle_deg]
    # x is p[:, 2], y is p[:, 3], angle_deg is p[:, 5]
    coords_list = [torch.stack([p[:, 2], p[:, 3], p[:, 5] / 360.0], dim=1) for p in points_list]
    
    # feats: [sin(ang), cos(ang), x, y, intensity_proc, angle_deg] (6-dim)
    feats_list = [p[:, :] for p in points_list]
    
    coords = torch.cat(coords_list, dim=0).contiguous()
    feats = torch.cat(feats_list, dim=0).contiguous()
    labels = torch.cat(labels_list, dim=0).contiguous()
    
    offsets = torch.tensor([p.shape[0] for p in points_list], dtype=torch.int).cumsum(0)
    
    lattice_labels = torch.stack([c['lattice'] for c in crystal_labels_list], dim=0)
    sg_labels = torch.cat([c['space_group'] for c in crystal_labels_list], dim=0).to(torch.long)

    batched_crystal_labels = {'lattice': lattice_labels.contiguous(), 'space_group': sg_labels.contiguous()}
    
    # Collect intensity statistics if available
    intensity_stats_batch = []
    for info in sample_info_list:
        if 'intensity_stats' in info and info['intensity_stats'] is not None:
            stats = info['intensity_stats']
            # Convert dict to tensor: [wilson, e2_std, cdf_1, cdf_2, cdf_3, gini, skewness, kurtosis, c_60, c_90, c_120, c_180]
            stats_tensor = torch.tensor([
                stats['wilson'], stats['e2_std'], stats['cdf_1'], stats['cdf_2'], stats['cdf_3'],
                stats['gini'], stats['skewness'], stats['kurtosis'],
                stats['c_60'], stats['c_90'], stats['c_120'], stats['c_180']
            ], dtype=torch.float32)
            intensity_stats_batch.append(stats_tensor)
        else:
            intensity_stats_batch.append(None)
    
    # Stack intensity stats if all samples have them
    if all(s is not None for s in intensity_stats_batch):
        intensity_stats = torch.stack(intensity_stats_batch, dim=0)
        batched_crystal_labels['intensity_stats'] = intensity_stats
    
    return coords, feats, labels, offsets, batched_crystal_labels, sample_info_list
