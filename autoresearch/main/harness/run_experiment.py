"""Independent experiment entrypoint: install the harness feedback instrument, then run train UNCHANGED.

This is how the two encoder loops launch an experiment -- through the harness, so the fast-feedback
signal (biological ``[profile] refined_seed_norm``; spacetime ``*_spacetime_gain`` deltas) lands in the
run log with NO edit to train.py / evaluate.py / fusion.py.

Usage (identical to train.py; the canonical evaluator always measures Earth4D gain):
  python -m deepearth.autoresearch.main.harness.run_experiment autoresearch/main/editable_files/champion.yaml --tag bio_maskw --cache_dir ...
"""
import sys

from deepearth.autoresearch.main.harness import hooks


def main():
    # Consume the old flag for command compatibility, but it no longer changes the benchmark suite.
    argv = [x for x in sys.argv[1:] if x != "--st-gain"]
    hooks.instrument()
    sys.argv = [sys.argv[0]] + argv          # hand the remaining args to train's argparse unchanged
    from deepearth.autoresearch.main.editable_files import train
    train.main()


if __name__ == "__main__":
    main()
