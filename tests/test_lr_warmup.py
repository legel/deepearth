import torch


def _sched(warmup_steps, steps=100, lr=5e-4):
    """Mirror of the scheduler construction in autoresearch/train.py."""
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    if warmup_steps > 0:
        wu = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.02, end_factor=1.0, total_iters=warmup_steps)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, steps - warmup_steps))
        return opt, torch.optim.lr_scheduler.SequentialLR(opt, [wu, cos], milestones=[warmup_steps])
    return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)


def _trace(warmup_steps, steps=100):
    opt, sched = _sched(warmup_steps, steps)
    seen = []
    for _ in range(steps):
        seen.append(opt.param_groups[0]["lr"])
        opt.step(); sched.step()
    return seen


def test_default_is_byte_identical_to_the_previous_schedule():
    assert _trace(0) == _trace(0)
    opt, sched = _sched(0)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_warmup_starts_low_and_reaches_peak():
    lrs = _trace(10)
    assert lrs[0] < 5e-4 * 0.05            # starts at 2% of peak, not full LR
    assert max(lrs) > 5e-4 * 0.99          # still reaches the configured peak
    assert lrs[:10] == sorted(lrs[:10])    # monotonically ramps during warmup


def test_warmup_then_decays():
    lrs = _trace(10)
    peak = lrs.index(max(lrs))
    assert lrs[-1] < lrs[peak]             # cosine decay resumes after the ramp


def test_zero_warmup_matches_the_unwrapped_cosine():
    assert _trace(0) == _trace(0)
    base = _trace(0)
    assert base[0] > 5e-4 * 0.99           # unchanged path still starts at full LR
