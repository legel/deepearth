"""Compatibility entrypoint for the production trainer."""

from deepearth.core.train import Experiment, main, train

__all__ = ["Experiment", "train"]


if __name__ == "__main__":
    main()
