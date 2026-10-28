"""Cooperative GPU pause: while the file PAUSE_GPU1 exists, a job waits at its next step boundary without computing.

The coordinator of the shared GPU creates the file to ask for the card and removes it to give it back. Jobs call
`wait_if_paused()` at safe points (between windows, every few training steps after a checkpoint). Cached GPU memory
is released while waiting.
"""

import time
from pathlib import Path

PAUSE_FILE = Path("/home/photon/4tb/fluxfill/model/PAUSE_GPU1")


def uses_gpu() -> bool:
    """False for CPU-only processes (CUDA_VISIBLE_DEVICES empty or no CUDA device), which the pause does not concern."""
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def wait_if_paused(tag: str = "", on_pause=None) -> bool:
    """Block while the pause file exists; return True if a pause happened. `on_pause` runs once before waiting.
    CPU-only processes never pause."""
    if not PAUSE_FILE.exists() or not uses_gpu():
        return False
    if on_pause is not None:
        on_pause()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    t0 = time.time()
    print(f"PAUSED {tag} at {time.strftime('%H:%M:%S')} ({PAUSE_FILE} exists)", flush=True)
    while PAUSE_FILE.exists():
        time.sleep(10)
    print(f"RESUMED {tag} at {time.strftime('%H:%M:%S')} after {time.time() - t0:.0f} s", flush=True)
    return True
