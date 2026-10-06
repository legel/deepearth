"""The Python port of usdm::vifstep keeps exactly the variables R keeps (same rows, threshold 5)."""
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from ranges.modelling import vifstep

R_ENV = "source ~/miniconda3/etc/profile.d/conda.sh && conda activate daru"


def _r_available() -> bool:
    if shutil.which("bash") is None:
        return False
    r = subprocess.run(["bash", "-lc", f"{R_ENV} && Rscript -e 'library(usdm)'"], capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not _r_available(), reason="R with usdm not available")
def test_vifstep_matches_usdm(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.normal(size=(3000, 6))
    cols = {f"v{i}": base[:, i] for i in range(6)}
    cols.update({f"c{i}": base[:, i] * (1 + 0.1 * i) + rng.normal(scale=0.05 + 0.1 * i, size=3000) for i in range(6)})
    df = pd.DataFrame(cols)
    f = tmp_path / "x.csv"
    df.to_csv(f, index=False)
    r = subprocess.run(["bash", "-lc", f"{R_ENV} && Rscript -e 'suppressMessages(library(usdm)); "
                        f"v <- vifstep(read.csv(\"{f}\"), th = 5); cat(v@results$Variables, sep = \"\\n\")'"],
                       capture_output=True, text=True, check=True)
    assert sorted(vifstep(df, 5.0)) == sorted(l for l in r.stdout.split("\n") if l.strip())
