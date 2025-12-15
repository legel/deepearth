"""
Training utilities for Earth4D encoder.
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


# =============================================================================
# YOHO Profiler
# =============================================================================

@dataclass
class YOHOProfilerStats:
    batch_size: int = 0
    total_time_ms: float = 0.0
    spatial_forward_ms: float = 0.0
    spatiotemporal_forward_ms: float = 0.0
    spatial_backward_ms: float = 0.0
    spatiotemporal_backward_ms: float = 0.0
    encoder_stats: Dict = None

    def __post_init__(self):
        if self.encoder_stats is None:
            self.encoder_stats = {}


class YOHOProfiler:
    """Profiler for YOHO optimization with dedup tracking."""

    def __init__(self):
        self.enabled = False
        self.batch_stats: List[YOHOProfilerStats] = []
        self._current: Optional[YOHOProfilerStats] = None
        self._timers: Dict[str, torch.cuda.Event] = {}
        self._full_forward_ms = 0.0
        self._full_backward_ms = 0.0

    def enable(self, enabled: bool = True):
        self.enabled = enabled
        if enabled:
            self.batch_stats = []

    def start_batch(self, batch_size: int):
        if not self.enabled:
            return
        self._current = YOHOProfilerStats(batch_size=batch_size)
        torch.cuda.synchronize()

    def start_timer(self, name: str):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self._timers[name] = torch.cuda.Event(enable_timing=True)
        self._timers[name].record()

    def stop_timer(self, name: str) -> float:
        if not self.enabled or name not in self._timers:
            return 0.0
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        torch.cuda.synchronize()
        return self._timers[name].elapsed_time(end)

    def record_encoder_dedup(self, encoder_name: str, batch_size: int, dedup_stats: Optional[Dict]):
        if not self._current:
            return
        stats = {'name': encoder_name, 'batch_size': batch_size, 'num_levels': 0, 'unique_per_level': None}
        if dedup_stats is not None:
            stats['num_levels'] = dedup_stats['num_levels']
            stats['unique_per_level'] = dedup_stats['unique_per_level']
        self._current.encoder_stats[encoder_name] = stats

    def record_spatial_forward(self, time_ms: float, batch_size: int, num_levels: int):
        if self._current:
            self._current.spatial_forward_ms = time_ms

    def record_spatiotemporal_forward(self, time_ms: float, batch_size: int, num_levels: int):
        if self._current:
            self._current.spatiotemporal_forward_ms = time_ms

    def end_batch(self):
        if self._current:
            self._current.total_time_ms = (
                self._current.spatial_forward_ms + self._current.spatiotemporal_forward_ms +
                self._current.spatial_backward_ms + self._current.spatiotemporal_backward_ms
            )
            self.batch_stats.append(self._current)
            self._current = None

    def get_epoch_summary(self) -> Dict:
        if not self.batch_stats:
            return {}
        encoder_totals = {}
        for batch in self.batch_stats:
            for enc_name, es in batch.encoder_stats.items():
                if enc_name not in encoder_totals:
                    encoder_totals[enc_name] = {'total_nominal': 0, 'total_actual': 0, 'num_levels': es['num_levels']}
                nominal = es['batch_size'] * es['num_levels'] * 8
                actual = int(es['unique_per_level'].sum()) * 8 if es['unique_per_level'] is not None else nominal
                encoder_totals[enc_name]['total_nominal'] += nominal
                encoder_totals[enc_name]['total_actual'] += actual
        return {
            'batches': len(self.batch_stats),
            'samples': sum(b.batch_size for b in self.batch_stats),
            'total_ms': sum(b.total_time_ms for b in self.batch_stats),
            'spatial_fwd_ms': sum(b.spatial_forward_ms for b in self.batch_stats),
            'spatiotemporal_fwd_ms': sum(b.spatiotemporal_forward_ms for b in self.batch_stats),
            'encoder_stats': encoder_totals,
        }

    def print_summary(self, epoch: int, yoho: bool):
        s = self.get_epoch_summary()
        if not s:
            return
        print(f"\n  ┌{'─'*76}┐")
        print(f"  │ YOHO PROFILER - Epoch {epoch} ({'YOHO ENABLED' if yoho else 'STANDARD'}){' '*36}│")
        print(f"  ├{'─'*76}┤")
        t = s['total_ms']
        if t > 0:
            print(f"  │ HASH ENCODER TIMING{' '*56}│")
            print(f"  │   Spatial Fwd:     {s['spatial_fwd_ms']:>8.1f}ms ({100*s['spatial_fwd_ms']/t:>5.1f}%){' '*29}│")
            print(f"  │   Spatiotemporal:  {s['spatiotemporal_fwd_ms']:>8.1f}ms ({100*s['spatiotemporal_fwd_ms']/t:>5.1f}%){' '*29}│")
        enc_stats = s.get('encoder_stats', {})
        if enc_stats:
            print(f"  ├{'─'*76}┤")
            print(f"  │ DEDUPLICATION{' '*63}│")
            total_nominal = total_actual = 0
            for enc_name in ['xyz', 'xyt', 'yzt', 'xzt']:
                if enc_name in enc_stats:
                    es = enc_stats[enc_name]
                    nominal, actual = es['total_nominal'], es['total_actual']
                    ratio = nominal / (actual + 1e-10)
                    savings = 100 * (1 - actual / (nominal + 1e-10))
                    total_nominal += nominal
                    total_actual += actual
                    print(f"  │   {enc_name:8s}  {nominal:>12,} nominal  {actual:>12,} actual  {ratio:>6.2f}x  {savings:>5.1f}%  │")
            if total_nominal > 0:
                overall_ratio = total_nominal / (total_actual + 1e-10)
                overall_savings = 100 * (1 - total_actual / (total_nominal + 1e-10))
                print(f"  │   {'TOTAL':8s}  {total_nominal:>12,} nominal  {total_actual:>12,} actual  {overall_ratio:>6.2f}x  {overall_savings:>5.1f}%  │")
        print(f"  └{'─'*76}┘")
        self.batch_stats = []


# =============================================================================
# Loss Computation
# =============================================================================

def compute_loss(predictions: torch.Tensor, targets: torch.Tensor, encoder,
                 criterion: Optional[Any] = None, enable_learned_probing: bool = False,
                 probe_entropy_weight: float = 0.5, enable_probe_entropy_loss: Optional[bool] = None,
                 enable_gradient_validation: bool = False) -> Dict:
    """Compute loss with optional entropy regularization for learned hash probing."""
    if criterion is None:
        criterion = torch.nn.MSELoss()
    if enable_probe_entropy_loss is None:
        enable_probe_entropy_loss = enable_learned_probing

    task_loss = criterion(predictions, targets)
    total_loss = task_loss
    loss_dict = {'task_loss': task_loss.item(), 'total_loss': task_loss.item()}

    if enable_probe_entropy_loss and hasattr(encoder, 'xyt_encoder') and \
       getattr(encoder.xyt_encoder, 'enable_learned_probing', False):
        entropy = _compute_probe_entropy(encoder)
        total_loss = total_loss - probe_entropy_weight * entropy
        loss_dict['probe_entropy_loss'] = entropy.item()
        loss_dict['total_loss'] = total_loss.item()

    loss_dict['_total_loss_tensor'] = total_loss
    return loss_dict


def _compute_probe_entropy(encoder) -> torch.Tensor:
    """Compute entropy of probe distributions."""
    total_entropy = 0.0
    num_encoders = 0
    for name in ['xyz_encoder', 'xyt_encoder', 'yzt_encoder', 'xzt_encoder']:
        enc = getattr(encoder, name, None)
        if enc is None or not getattr(enc, 'enable_learned_probing', False):
            continue
        if not hasattr(enc, 'index_logits') or enc.index_logits is None:
            continue
        probs = torch.softmax(enc.index_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        total_entropy = total_entropy + entropy
        num_encoders += 1
    if num_encoders == 0:
        return torch.tensor(0.0, device=encoder.xyz_encoder.embeddings.device)
    return total_entropy / num_encoders


# =============================================================================
# Reporting
# =============================================================================

def print_resolution_info(encoder, config: Dict[str, Any], adaptive_range: Optional[Any] = None):
    """Print detailed resolution information for Earth4D encoder."""
    results = _calculate_resolution_scales(encoder)

    print("\n" + "="*80)
    print("EARTH4D INITIALIZATION REPORT")
    print("="*80)

    print("\n┌─ ENHANCEMENT CONFIGURATION ─────────────────────────────────────────────┐")
    print(f"│  Adaptive Range:     {'ENABLED' if config.get('use_adaptive_range') else 'disabled':12}                                  │")
    print(f"│  YOHO Optimization:  {'ENABLED' if config.get('use_yoho') else 'disabled':12}                                  │")
    lp_str = f"ENABLED (N_p={config.get('probing_range', 0)})" if config.get('enable_learned_probing') else 'disabled'
    print(f"│  Learned Probing:    {lp_str:24}              │")
    print("└─────────────────────────────────────────────────────────────────────────┘")

    print("\n" + "-"*80)
    print("RESOLUTION SCALE TABLE")
    print("-"*80)

    effective_multiplier = 1.0
    if config.get('use_adaptive_range') and adaptive_range is not None:
        coverage = adaptive_range.get_coordinate_coverage()
        avg_coverage = sum(coverage.values()) / 3
        if avg_coverage > 0:
            effective_multiplier = 1.0 / avg_coverage

    print("\nSPATIAL ENCODER (XYZ):")
    print(f"{'Level':<6} {'Grid Res':<12} {'Meters/Cell':<15} {'KM/Cell':<12}")
    print("-" * 70)
    for item in results['spatial']:
        meters = item['meters_per_cell']
        if meters >= 1000:
            meters_str = f"{meters/1000:.1f}km"
        elif meters >= 1:
            meters_str = f"{meters:.2f}m"
        else:
            meters_str = f"{meters:.3f}m"
        km_str = f"{item['km_per_cell']:.3f}" if item['km_per_cell'] < 1 else f"{item['km_per_cell']:.2f}"
        print(f"{item['level']:<6} {item['grid_resolution']:<12} {meters_str:<15} {km_str:<12}")

    print("\nSPATIOTEMPORAL ENCODERS (XYT, YZT, XZT):")
    print(f"{'Level':<6} {'Grid Res':<12} {'Seconds/Cell':<15} {'Days/Cell':<12}")
    print("-" * 70)
    for item in results['temporal']['xyt']:
        print(f"{item['level']:<6} {item['grid_resolution']:<12} {item['seconds_per_cell']:<15.1f} {item['days_per_cell']:<12.2f}")

    spatial_params = encoder.xyz_encoder.embeddings.numel()
    temporal_params = sum(getattr(encoder, f'{n}_encoder').embeddings.numel() for n in ['xyt', 'yzt', 'xzt'])
    total_params = spatial_params + temporal_params
    total_memory = total_params * 4 / (1024 * 1024)

    spatial_hash_entries = 2 ** encoder.spatial_log2_hashmap_size
    temporal_hash_entries = 2 ** encoder.temporal_log2_hashmap_size

    print(f"\nHASH TABLE CONFIGURATION:")
    print(f"  Spatial: 2^{encoder.spatial_log2_hashmap_size} = {spatial_hash_entries:,} entries")
    print(f"  Spatiotemporal: 2^{encoder.temporal_log2_hashmap_size} = {temporal_hash_entries:,} entries")
    print(f"  Total capacity: {spatial_hash_entries + temporal_hash_entries*3:,} entries")

    print(f"\nACTUAL PARAMETERS (MEMORY FOOTPRINT):")
    print(f"  Spatial encoders: {spatial_params:,} params = {spatial_params * 4 / (1024*1024):.2f} MB")
    print(f"  Spatiotemporal encoders: {temporal_params:,} params = {temporal_params * 4 / (1024*1024):.2f} MB")
    print(f"  Total: {total_params:,} params = {total_memory:.2f} MB")
    print(f"  During training (4x): ~{total_memory * 4:.2f} MB")


def _calculate_resolution_scales(encoder) -> Dict:
    """Calculate resolution scales for all encoders."""
    earth_radius = 6371000.0
    physical_range = 2 * earth_radius
    results = {'spatial': [], 'temporal': {'xyt': [], 'yzt': [], 'xzt': []}}

    spatial_encoder = encoder.xyz_encoder
    for level in range(spatial_encoder.num_levels):
        base_res = spatial_encoder.base_resolution[0].item()
        scale = spatial_encoder.per_level_scale[0].item()
        grid_resolution = np.ceil(base_res * (scale ** level))
        meters_per_cell = physical_range / grid_resolution
        results['spatial'].append({
            'level': level, 'grid_resolution': int(grid_resolution),
            'meters_per_cell': meters_per_cell, 'km_per_cell': meters_per_cell / 1000
        })

    seconds_per_year = 365.25 * 24 * 3600
    for name, enc in [('xyt', encoder.xyt_encoder), ('yzt', encoder.yzt_encoder), ('xzt', encoder.xzt_encoder)]:
        for level in range(enc.num_levels):
            base_res = enc.base_resolution[0].item()
            scale = enc.per_level_scale[0].item()
            grid_resolution = np.ceil(base_res * (scale ** level))
            seconds_per_cell = seconds_per_year / grid_resolution
            results['temporal'][name].append({
                'level': level, 'grid_resolution': int(grid_resolution),
                'seconds_per_cell': seconds_per_cell, 'days_per_cell': seconds_per_cell / 86400
            })
    return results
