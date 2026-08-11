"""Training run capture. One JSONL file per run, flushed per event for live tailing.

    from dashboard.logger import RunLogger
    log = RunLogger("compact-25m", config=cfg_dict)
    log.event("step", step=i, loss=float(loss))
    log.event("eval", step=i, scores=scores)
    log.final(scores)
"""
import json, os, time
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"


class RunLogger:
    def __init__(self, name: str, config: dict | None = None):
        RUNS.mkdir(exist_ok=True)
        self.id = f"{time.strftime('%Y%m%d_%H%M%S')}_{name}"
        self.path = RUNS / f"{self.id}.jsonl"
        self._f = open(self.path, "a")
        self.event("config", name=name, config=config or {}, pid=os.getpid())

    def event(self, t: str, **kw):
        self._f.write(json.dumps({"t": t, "ts": round(time.time(), 3), **kw}) + "\n")
        self._f.flush()

    def final(self, scores: dict, **kw):
        self.event("final", scores=scores, **kw)
        self._f.close()
