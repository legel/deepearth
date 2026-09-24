"""Figures from a published product: the wake and the vortices a reader judges by eye.

    python3 docs/make_figures.py outputs/campanile/wind_tower_0.8m.bin

Vorticity is signed, so it gets a diverging map centred on zero; speed gets Turbo, the ramp
the viewer uses for water and wind.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frames  # noqa: E402

DOCS = Path(__file__).resolve().parent


def _levels(header: frames.Header) -> np.ndarray:
    zf = np.asarray(header.zf)
    return 0.5 * (zf[:-1] + zf[1:])


def plan(path: Path, height_m: float, frame: int = 0) -> Path:
    """Speed and vertical vorticity on a horizontal slice, side by side."""
    header, _, data = frames.read_fields(path, frame=frame)
    zc = _levels(header)
    k = int(np.argmin(np.abs(zc - height_m)))
    u, v, w = data[0, 0, k], data[0, 1, k], data[0, 2, k]
    wz = data[0, 5, k]
    speed = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    extent = [0, header.shape[2] * header.cell_m, 0, header.shape[1] * header.cell_m]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    im = axes[0].imshow(speed, origin="lower", extent=extent, cmap="turbo", vmin=0)
    fig.colorbar(im, ax=axes[0], label="speed [m/s]")
    axes[0].set_title(f"speed, z = {zc[k]:.1f} m")
    cap = float(np.percentile(np.abs(wz), 99.5))
    im = axes[1].imshow(wz, origin="lower", extent=extent, cmap="RdBu_r", vmin=-cap, vmax=cap)
    fig.colorbar(im, ax=axes[1], label="vertical vorticity [1/s]")
    axes[1].set_title(f"vorticity, z = {zc[k]:.1f} m")
    step = max(1, header.shape[2] // 40)
    x = (np.arange(header.shape[2]) + 0.5) * header.cell_m
    y = (np.arange(header.shape[1]) + 0.5) * header.cell_m
    for ax in axes:
        ax.quiver(x[::step], y[::step], u[::step, ::step], v[::step, ::step],
                  color="k", alpha=0.35, scale=None)
        ax.set_xlabel("east [m]")
    axes[0].set_ylabel("north [m]")
    out = DOCS / f"{path.stem}_plan_{zc[k]:.0f}m.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def section(path: Path, frame: int = 0) -> Path:
    """Speed and cross-stream vorticity on the vertical plane through the domain centre."""
    header, _, data = frames.read_fields(path, frame=frame)
    zc = _levels(header)
    j = header.shape[1] // 2
    u, v, w = data[0, 0, :, j], data[0, 1, :, j], data[0, 2, :, j]
    wy = data[0, 4, :, j]
    speed = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    x = (np.arange(header.shape[2]) + 0.5) * header.cell_m

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True, sharex=True)
    im = axes[0].pcolormesh(x, zc, speed, cmap="turbo", vmin=0, shading="nearest")
    fig.colorbar(im, ax=axes[0], label="speed [m/s]")
    axes[0].set_title("speed on the centre section")
    cap = float(np.percentile(np.abs(wy), 99.5))
    im = axes[1].pcolormesh(x, zc, wy, cmap="RdBu_r", vmin=-cap, vmax=cap, shading="nearest")
    fig.colorbar(im, ax=axes[1], label="cross-stream vorticity [1/s]")
    axes[1].set_title("vorticity on the centre section")
    for ax in axes:
        ax.set_ylabel("height [m]")
    axes[1].set_xlabel("east [m]")
    out = DOCS / f"{path.stem}_section.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    path = Path(sys.argv[1])
    header = frames.read_fields(path, frame=0)[0]
    heights = [float(a) for a in sys.argv[2:]] or [0.5 * float(header.zf[-1]) / 2]
    for h in heights:
        print(plan(path, h))
    print(section(path))


if __name__ == "__main__":
    main()
