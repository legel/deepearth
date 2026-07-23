"""Independent experiment entrypoint: install the programs/ feedback instrument, then run train UNCHANGED.

This is how the two encoder loops launch an experiment -- through programs/, so the fast-feedback signal
(biological ``[profile] refined_seed_norm``; spacetime ``*_spacetime_gain`` deltas) lands in the run log
with NO edit to train.py / evaluate.py / fusion.py.

Usage (identical to train.py, plus optional --st-gain to build st_gain via the spacetime ablation):
  python -m deepearth.autoresearch.programs.run_experiment autoresearch/champion.yaml --tag bio_maskw --cache_dir ... [--st-gain]
"""
import sys

from deepearth.autoresearch.programs import hooks


def main():
    argv = [x for x in sys.argv[1:] if x != "--st-gain"]
    hooks.instrument(spacetime_gain=("--st-gain" in sys.argv))
    sys.argv = [sys.argv[0]] + argv          # hand the remaining args to train's argparse unchanged
    from deepearth.autoresearch import train
    train.main()


if __name__ == "__main__":
    main()
