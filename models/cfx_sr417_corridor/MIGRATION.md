# Migrating cfx_sr417_corridor to ASU Sol

Written 2026-07-23. Goal: move this project to Sol's `/scratch` storage to get a real
Python 3.10+ environment and an actual CUDA GPU (solves the physicsnemo Python-version
blocker and gives a real, not MPS-only, GPU benchmark for the compute-strategy work).

**No data needs re-downloading.** Everything already fetched/derived on the Mac (DEM,
LiDAR, SSURGO, NAIP, simulation outputs, etc.) transfers as plain files via `rsync`. Only
the *Python environment* is rebuilt, since pip wheels are platform-specific (Mac arm64 vs.
Sol's x86_64 Linux) — that's a normal `pip install`, not redone work.

Steps below are things **you** run (VPN + SSH need your interactive login /
Duo 2FA, which I can't do from here). Where I'm not 100% certain of Sol's current
partition/module names, I've flagged it — confirm with `sinfo` / `module avail` /
ASU Research Computing docs rather than trusting my guess blindly, since HPC cluster
configs change over time.

## 1. Connect

```bash
# Connect your usual ASU VPN client first, then:
ssh <your_asurite>@sol.asu.edu
```

## 2. Create the project location under scratch

```bash
mkdir -p /scratch/$USER/deepearth/models
```

(Confirm `/scratch/$USER` is really your scratch root — some HPC configs use a
different convention; `echo $SCRATCH` or ASU RC's docs will confirm it if this path
doesn't exist.)

## 3. Transfer the project (run this FROM THE MAC, not from Sol, VPN connected)

```bash
rsync -avz --progress \
  /Users/hqqq422/Desktop/deepearth/models/cfx_sr417_corridor/ \
  <your_asurite>@sol.asu.edu:/scratch/<your_asurite>/deepearth/models/cfx_sr417_corridor/
```

Check total size first — `du -sh cfx_sr417_corridor/` — before transferring. If scratch
quota is tight, the biggest single chunk is the raw LiDAR tiles:

```bash
du -sh lidar/data/raw/    # ~1.57 GB, LAZ tiles
```

Those ARE cheaply re-fetchable on Sol (`python3 lidar/build_lidar_pointcloud.py`
re-downloads them from the USGS TNM API) if you'd rather exclude them from the
transfer and save the bandwidth/quota:

```bash
rsync -avz --progress --exclude 'lidar/data/raw/' \
  /Users/hqqq422/Desktop/deepearth/models/cfx_sr417_corridor/ \
  <your_asurite>@sol.asu.edu:/scratch/<your_asurite>/deepearth/models/cfx_sr417_corridor/
```

Everything else (derived DEM/terrain/simulation outputs) is real prior work — don't
exclude those, they're not cheaply reproducible (some runs took 20+ minutes).

## 4. Python environment on Sol

```bash
cd /scratch/$USER/deepearth/models/cfx_sr417_corridor
module avail mamba   # or `module avail conda` / `module avail python` — confirm exact name
module load mamba/latest   # adjust to whatever `module avail` actually shows

# Use an explicit scratch PATH, not a named env (-n) — named envs default to installing
# into $HOME/.conda/envs, which works against the whole point of moving to scratch for
# storage. A -p env is otherwise identical (isolated, doesn't touch other projects/envs).
mamba create -p /scratch/$USER/envs/cfx python=3.11 -y
mamba activate /scratch/$USER/envs/cfx

# Check Sol's CUDA module version before installing torch:
module avail cuda
module load cuda/<version shown above>
pip install torch torch_geometric torch_scatter \
  --index-url https://download.pytorch.org/whl/cu121   # match the CUDA version you loaded

pip install -r requirements.txt   # everything else (rasterio, geopandas, pysheds, etc.)
pip install nvidia-physicsnemo    # only works now because Python is 3.10+ — this failed
                                   # on the Mac's stock Python 3.9.6
```

## 5. GPU allocation (Sol uses SLURM — you can't just run GPU code on the login node)

Interactive test session:

```bash
srun -p general -q public --gres=gpu:a100:1 -c 8 --mem=32G -t 02:00:00 --pty bash
```

(Partition/QOS/GPU-type names — `general`/`public`/`a100` — are my best recollection of
Sol's typical setup, not a live-confirmed fact. Run `sinfo` once logged in to see actual
partition/GPU names before trusting this command.)

For longer runs, an `sbatch` script is better than holding an interactive shell open —
ask me to draft one once you know which job (the torch GPU solver benchmark, or a
HydroGraphNet training run) you want to submit first.

## 6. Install Claude Code on Sol, so you can keep working there directly

```bash
module avail node   # or nvm if no module exists
npm install -g @anthropic-ai/claude-code
cd /scratch/$USER/deepearth/models/cfx_sr417_corridor
claude
```

**Important: this starts a brand-new session, not a continuation of this one.** It has
no memory of this conversation. But since `CLAUDE.md` in this project directory travels
with the rsync, the new session will read it automatically and immediately have almost
all the context from this project's history — that's the intended continuity mechanism,
not something you need to re-explain by hand. The one thing that *won't* transfer is the
cross-session "auto memory" notes tied to this Mac's local Claude Code install
(collaboration-style preferences, mostly) — minor, not technical content, not worth
re-deriving.
