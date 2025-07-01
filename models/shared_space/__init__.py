"""
Shared-space multimodal fusion package.

Importing either of the items below pulls in the full
`multimodal_shared_space` implementation so you can write:

    from models.shared_space import build_model
    model = build_model()

or

    from models.shared_space import MultiModalSharedSpace
"""

from .multimodal_shared_space import (
    build_model,
    MultiModalSharedSpace,
)

__all__: list[str] = ["build_model", "MultiModalSharedSpace"]

