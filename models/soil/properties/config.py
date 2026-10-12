"""Where the soil-property model keeps its data: DEEPEARTH_SOIL_DATA (default ~/.deepearth/soil). Benchmark inputs
(NEON, WoSIS), sampled points, metrics and the fitted site model live under it."""
import os
import pathlib

DATA = pathlib.Path(os.environ.get("DEEPEARTH_SOIL_DATA", pathlib.Path.home() / ".deepearth" / "soil"))
