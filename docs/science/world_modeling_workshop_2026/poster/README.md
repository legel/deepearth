# DeepEarth Poster - 2026 World Modeling Workshop

Poster #70 for the World Modeling Workshop at Mila, Quebec AI Institute.

## Build Commands

```bash
# Step 1 & 2: Compile LaTeX (2 passes for references)
xelatex -interaction=nonstopmode deepearth_poster.tex
xelatex -interaction=nonstopmode deepearth_poster.tex

# Step 3: Compress PDF with Ghostscript
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -dNOPAUSE -dQUIET -dBATCH \
   -dEmbedAllFonts=true -dSubsetFonts=true \
   -sOutputFile=deepearth_poster_compressed.pdf deepearth_poster.pdf

# Step 4: Pad to 36x48 inches (2592x3456 pts) with centered content
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -dNOPAUSE -dQUIET -dBATCH \
   -dDEVICEWIDTHPOINTS=2592 -dDEVICEHEIGHTPOINTS=3456 \
   -dFIXEDMEDIA -dPDFFitPage \
   -dEmbedAllFonts=true -dSubsetFonts=true \
   -sOutputFile=70_36x48.pdf deepearth_poster_compressed.pdf

# Step 5: Copy to descriptive filename
cp 70_36x48.pdf DeepEarth_2026_World_Modeling_Workshop_Poster.pdf

# Verify dimensions (should show 2592 x 3456 pts = 36" x 48")
pdfinfo 70_36x48.pdf | grep "Page size"
```

## Requirements

- XeLaTeX (TeX Live 2025)
- Ghostscript
- Oxygen font family (Regular, Bold, Light) in `~/Library/Fonts/`
- Menlo font (macOS system font)

## Input Files

- `deepearth_poster.tex` - Main LaTeX source
- `wmw2026_poster_header.pdf` - Custom header with title, authors, and logos
- `../figures/` - Figures directory (deepearth.pdf, earth4d.pdf, etc.)

## Output Files

- `70_36x48.pdf` - Final poster (36" x 48")
- `DeepEarth_2026_World_Modeling_Workshop_Poster.pdf` - Same file, descriptive name
