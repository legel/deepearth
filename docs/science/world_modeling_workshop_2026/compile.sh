#!/bin/bash

# Compilation script for DeepEarth World Modeling Workshop 2026 paper
# Uses pdfLaTeX for consistent layout

echo "Starting compilation of deepearth.tex..."

# First pass
echo "[1/4] Running pdfLaTeX (first pass)..."
pdflatex -interaction=nonstopmode deepearth.tex > /tmp/deepearth_pass1.log 2>&1

if [ $? -ne 0 ]; then
    echo "ERROR: First pdfLaTeX pass failed!"
    echo "Check /tmp/deepearth_pass1.log for details"
    exit 1
fi

# BibTeX
echo "[2/4] Running BibTeX..."
bibtex deepearth > /tmp/deepearth_bibtex.log 2>&1

if [ $? -ne 0 ]; then
    echo "WARNING: BibTeX reported errors (this may be okay)"
    echo "Check /tmp/deepearth_bibtex.log for details"
fi

# Second pass
echo "[3/4] Running pdfLaTeX (second pass)..."
pdflatex -interaction=nonstopmode deepearth.tex > /tmp/deepearth_pass2.log 2>&1

if [ $? -ne 0 ]; then
    echo "ERROR: Second pdfLaTeX pass failed!"
    echo "Check /tmp/deepearth_pass2.log for details"
    exit 1
fi

# Third pass
echo "[4/4] Running pdfLaTeX (final pass)..."
pdflatex -interaction=nonstopmode deepearth.tex > /tmp/deepearth_pass3.log 2>&1

if [ $? -ne 0 ]; then
    echo "ERROR: Final pdfLaTeX pass failed!"
    echo "Check /tmp/deepearth_pass3.log for details"
    exit 1
fi

# Check if Ghostscript is available for compression
if command -v gs &> /dev/null; then
    # Rename output for compression
    mv deepearth.pdf deepearth_uncompressed.pdf 2>/dev/null

    # Get original file size
    ORIGINAL_SIZE=$(du -h deepearth_uncompressed.pdf | awk '{print $1}')

    # Compress PDF without losing resolution using Ghostscript
    echo "[5/5] Compressing PDF (lossless)..."
    gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -dNOPAUSE -dQUIET -dBATCH \
       -dPrinted=false \
       -sOutputFile=deepearth.pdf \
       deepearth_uncompressed.pdf > /tmp/deepearth_compress.log 2>&1

    if [ $? -ne 0 ]; then
        echo "WARNING: PDF compression failed, using uncompressed version"
        mv deepearth_uncompressed.pdf deepearth.pdf
    else
        # Get compressed file size
        COMPRESSED_SIZE=$(du -h deepearth.pdf | awk '{print $1}')
        echo "PDF compressed: $ORIGINAL_SIZE -> $COMPRESSED_SIZE"
        # Remove uncompressed version
        rm deepearth_uncompressed.pdf 2>/dev/null
    fi
else
    echo "[5/5] Ghostscript not found, skipping compression..."
fi

# Check page count if pdfinfo is available
if command -v pdfinfo &> /dev/null; then
    PAGES=$(pdfinfo deepearth.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}')
    echo ""
    echo "=========================================="
    echo "Compilation successful!"
    echo "=========================================="
    echo "Output: deepearth.pdf"
    echo "Pages: $PAGES"
else
    echo ""
    echo "=========================================="
    echo "Compilation successful!"
    echo "=========================================="
    echo "Output: deepearth.pdf"
fi

echo ""
echo "Log files saved to /tmp/deepearth_*.log"
echo ""
