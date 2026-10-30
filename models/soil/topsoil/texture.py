"""USDA soil texture class from sand and clay percent (silt = 100 - sand - clay), the 12 classes of the USDA texture
triangle (Soil Survey Manual, 2017, Fig. 3-7). Vectorised; returns class names, NaN-safe ("" where unknown).
This is the SOIL ▸ TYPE layer: a gardener's vocabulary (sandy loam, clay loam ...) for the same two numbers."""
import numpy as np

CLASSES = ("sand", "loamy sand", "sandy loam", "loam", "silt loam", "silt", "sandy clay loam", "clay loam",
           "silty clay loam", "sandy clay", "silty clay", "clay")


def texture_class(sand, clay):
    sand, clay = np.asarray(sand, float), np.asarray(clay, float)
    silt = 100 - sand - clay
    out = np.full(sand.shape, "", dtype=object)
    rules = [  # order matters: first match wins, following the triangle's boundary lines
        ("sand", (silt + 1.5 * clay) < 15),
        ("loamy sand", ((silt + 1.5 * clay) >= 15) & ((silt + 2 * clay) < 30)),
        ("sandy loam", ((clay >= 7) & (clay < 20) & (sand > 52) & ((silt + 2 * clay) >= 30)) |
                       ((clay < 7) & (silt < 50) & ((silt + 2 * clay) >= 30))),
        ("loam", (clay >= 7) & (clay < 27) & (silt >= 28) & (silt < 50) & (sand <= 52)),
        ("silt loam", ((silt >= 50) & (clay >= 12) & (clay < 27)) | ((silt >= 50) & (silt < 80) & (clay < 12))),
        ("silt", (silt >= 80) & (clay < 12)),
        ("sandy clay loam", (clay >= 20) & (clay < 35) & (silt < 28) & (sand > 45)),
        ("clay loam", (clay >= 27) & (clay < 40) & (sand > 20) & (sand <= 45)),
        ("silty clay loam", (clay >= 27) & (clay < 40) & (sand <= 20)),
        ("sandy clay", (clay >= 35) & (sand > 45)),
        ("silty clay", (clay >= 40) & (silt >= 40)),
        ("clay", (clay >= 40) & (sand <= 45) & (silt < 40)),
    ]
    ok = ~(np.isnan(sand) | np.isnan(clay))
    for name, m in rules:
        sel = ok & m & (out == "")
        out[sel] = name
    return out


if __name__ == "__main__":
    # textbook corners of the triangle
    tests = {(92, 3): "sand", (82, 5): "loamy sand", (65, 10): "sandy loam", (40, 20): "loam", (20, 15): "silt loam",
             (5, 5): "silt", (60, 25): "sandy clay loam", (32, 33): "clay loam", (10, 33): "silty clay loam",
             (50, 40): "sandy clay", (5, 45): "silty clay", (20, 60): "clay", (43, 24.5): "loam"}
    got = texture_class([k[0] for k in tests], [k[1] for k in tests])
    bad = [(k, v, g) for (k, v), g in zip(tests.items(), got) if v != g]
    print("all textbook points correct" if not bad else bad)
