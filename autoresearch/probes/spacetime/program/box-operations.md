# Earth4D Agent — Box & Operations (LOCAL / gitignored — contains box connection details)

Operational runbook for running the Earth4D agentic loop. The science/loop doctrine is in `program.md`;
this is the "how to actually run it on the box" layer. **Not committed** (holds SSH host + connection details).

## Box
- **GPU box:** `ssh newbox` = `root@222.228.49.105 -p 30474` (key `~/.ssh/id_ed25519`; `StrictHostKeyChecking accept-new`, `ServerAliveInterval 15`).
- Strip the login banner from output: `... 2>&1 | grep -vE "vast.ai|Welcome|Have fun|AI agents|READ /etc"`.
- It is a **vast.ai container** — a "reboot" is a container restart, NOT a host GPU reset.
- **Run dir:** `/workspace/deepearth` with `PYTHONPATH=/workspace`.
- **Edit + commit** in the local clone `/Users/andromeda/deepcal-archive/deepearth`, then `scp` changed files to the box. Remote branch: `origin/deepcal-ensue-autoresearch`.

## First thing, every session
1. **Restore the cache symlink** (the container restart drops it; without it every probe dies with `gbif_vocab.npz not found`):
   `[ -e /workspace/data ] || ln -s /workspace/deepearth/data /workspace/data`
2. **Check each GPU with a LIVE CUDA op — never trust the `nvidia-smi` util counter** (it shows a phantom 100% that does NOT mean wedged):
   `CUDA_VISIBLE_DEVICES=<i> timeout 10 python3.12 -c "import torch;print((torch.randn(999,999,device='cuda')@torch.randn(999,999,device='cuda')).sum().item())"`
   → returns a number ⇒ usable; hangs/errors ⇒ genuinely wedged, skip that GPU (needs a host reset a container reboot won't give).

## Running experiments
- No background driver. `loop.sh`/`start.sh` were removed (random 3-knob sampler, max-of-N records, hogged both GPUs).
- Run one deliberate swing at a time:
  `cd /workspace/deepearth && PYTHONPATH=/workspace python3.12 -m deepearth.autoresearch.probes.spacetime.editable_files.harness --metric <cap> --tag <id> --device cuda:N --ensue`
- Check nothing is holding a GPU first: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
  (empty = free; the util% counter shows a phantom 98% and means nothing).
- Killing stragglers: use a self-excluding pattern — `pgrep -f "spacetime[.]probe" | xargs -r kill -9`.
  A bare `pkill -f "spacetime.probe"` matches its OWN command line and kills your shell before it acts.

## Ensue + commits
- **Ensue token:** `/workspace/.env` (`ENSUE_API_TOKEN`); `harness.py` reads it automatically. **NEVER commit it.**
- Each run upserts `LOOP-earth4d-<capability>` (best + record-history + dead-ends-with-reason). See `program.md` §1.6.
- **Commit identity:** `git -c user.name='Sai Vegasena' -c user.email='saidcooldude@gmail.com' commit ...` — **no co-author line.**

## Report each check-in
Iterations since last, any new records (old→new + receipt), and the full current-best scorecard:
`python3 -c "import json;d=json.load(open('autoresearch/probes/spacetime/records/records.json'));[print(k,round(v['score'],4),v.get('tag')) for k,v in sorted(d.items(),key=lambda x:-x[1]['score'])]"`
