# DeepEarth: Self-Supervised Multi-Modal World Model with 4D Space-Time Embedding

This directory contains the LaTeX source files for the DeepEarth paper accepted to the **2026 World Modeling Workshop** at the **Mila - Quebec AI Institute**.

## Paper Overview

We present **DeepEarth**, a self-supervised multi-modal world model with **Earth4D**, a novel planetary-scale 4D space-time positional encoder. Earth4D extends 3D multi-resolution hash encoding to include time, efficiently scaling across the planet over centuries with sub-meter, sub-second precision.

**Key Results:**
- Earth4D achieves state-of-the-art performance on the Globe-LFMC 2.0 ecological forecasting benchmark
- MAE 12.1pp and R² 0.755, surpassing pre-trained foundation models
- Uses only (x,y,z,t) coordinates and species embeddings—no satellite imagery, weather data, or topography required

## Directory Structure

```
world_modeling_workshop_2026/
├── README.md                    # This file
├── compile.sh                   # Compilation script
├── deepearth.tex                # Main LaTeX document
├── deepearth.bib                # Bibliography file
├── wmw2026_conference.sty       # Conference style file
├── wmw2026_conference.bst       # Bibliography style file
├── math_commands.tex            # Math notation macros
├── natbib.sty                   # Citation package
├── fancyhdr.sty                 # Header/footer package
└── figures/
    ├── deepearth.pdf            # DeepEarth architecture diagram
    ├── earth4d.pdf              # Earth4D encoder diagram
    ├── error_distribution_histogram.png   # LFMC error distribution
    ├── geospatial_temporal_test.png       # Geographic/temporal results
    ├── earth4d_resolution_levels.png      # Resolution specifications
    ├── 1M_hash_collision_rate.png         # Hash collision analysis
    └── rgb_reconstruction.png             # RGB reconstruction results
```

## Prerequisites

### macOS Installation

**Option 1: Using Homebrew (Recommended)**

```bash
# Install MacTeX (full TeX Live distribution)
brew install --cask mactex

# Install Ghostscript for PDF compression (optional)
brew install ghostscript

# Install poppler for pdfinfo (optional, for page count)
brew install poppler
```

**Option 2: Direct Download**

1. Download MacTeX from [tug.org/mactex](https://www.tug.org/mactex/mactex-download.html)
2. Double-click the `.pkg` file and follow the installation instructions
3. Installation takes approximately 10 minutes

MacTeX-2025 requires macOS 10.14 (Mojave) or higher and runs natively on Intel and Apple Silicon processors.

**Verify Installation:**

```bash
which pdflatex
pdflatex --version
```

If `pdflatex` is not found, add TeX Live to your PATH:

```bash
# For zsh (default on modern macOS)
echo 'export PATH=/usr/local/texlive/2025/bin/universal-darwin:$PATH' >> ~/.zshrc
source ~/.zshrc

# For bash
echo 'export PATH=/usr/local/texlive/2025/bin/universal-darwin:$PATH' >> ~/.bash_profile
source ~/.bash_profile
```

### Linux Installation (Ubuntu/Debian)

**Option 1: Using apt (Recommended for simplicity)**

```bash
# Basic installation
sudo apt-get update
sudo apt-get install texlive-latex-extra

# Install recommended fonts
sudo apt-get install texlive-fonts-recommended texlive-fonts-extra

# Install Ghostscript for PDF compression (optional)
sudo apt-get install ghostscript

# Install poppler-utils for pdfinfo (optional)
sudo apt-get install poppler-utils
```

For a complete installation with all packages:

```bash
sudo apt-get install texlive-full
```

**Option 2: Direct Installation from TeX Users Group**

```bash
# Download the installer
wget http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
tar -xzf install-tl-unx.tar.gz
cd install-tl-*

# Run the installer (requires sudo for system-wide installation)
sudo ./install-tl
```

After installation, add TeX Live to your PATH:

```bash
echo 'export PATH=/usr/local/texlive/2025/bin/x86_64-linux:$PATH' >> ~/.bashrc
source ~/.bashrc
```

**Verify Installation:**

```bash
pdflatex --version
bibtex --version
```

## Compilation

### Quick Start

```bash
cd world_modeling_workshop_2026
./compile.sh
```

This will:
1. Run pdfLaTeX (first pass)
2. Run BibTeX for bibliography processing
3. Run pdfLaTeX (second pass for cross-references)
4. Run pdfLaTeX (final pass)
5. Compress the PDF using Ghostscript (if available)

The output will be `deepearth.pdf`.

### Manual Compilation

If you prefer to run the commands manually:

```bash
pdflatex deepearth.tex
bibtex deepearth
pdflatex deepearth.tex
pdflatex deepearth.tex
```

### Troubleshooting

**"pdflatex: command not found"**
- Ensure TeX Live is installed and in your PATH (see installation instructions above)

**Missing packages**
- On macOS with MacTeX, all packages should be included
- On Linux, install `texlive-latex-extra` for additional packages

**BibTeX warnings**
- Warnings about undefined citations on first run are normal; they resolve after the full compilation cycle

**Log files**
- Compilation logs are saved to `/tmp/deepearth_*.log`
- Check these files for detailed error messages if compilation fails

## Citation

```bibtex
@inproceedings{legel2026deepearth,
  title={Self-Supervised Multi-Modal World Model with 4D Space-Time Embedding},
  author={Legel, Lance and Huang, Qin and Voelker, Brandon and Neamati, Daniel and
          Johnson, Patrick Alan and Bastani, Favyen and Rose, Jeff and
          Hennessy, James Ryan and Guralnick, Robert and Soltis, Douglas and
          Soltis, Pamela and Wang, Shaowen},
  booktitle={2026 World Modeling Workshop at Mila - Quebec AI Institute},
  year={2026}
}
```

## Related Links

- **DeepEarth Repository:** [github.com/legel/deepearth](https://github.com/legel/deepearth)
- **Earth4D Encoder:** [github.com/legel/deepearth/tree/main/encoders/xyzt](https://github.com/legel/deepearth/tree/main/encoders/xyzt)
- **World Modeling Workshop:** [world-model-mila.github.io](https://world-model-mila.github.io/)

## License

This paper and its LaTeX source are part of the DeepEarth open source project. See the main repository for license details.
