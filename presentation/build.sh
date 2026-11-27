#!/usr/bin/env bash
# Builds the deck into Final_Presentation.pdf. The extra LaTeX packages the deck needs (carlito, fontaxes, mweights,
# environ, trimspaces, downloaded from CTAN) are in ./texmf, so nothing is installed in the home folder: TEXMFHOME
# points there and the Carlito font map is loaded on the command line.
#   usage (from this folder):  ./build.sh
set -euo pipefail
cd "$(dirname "$0")"
export TEXMFHOME="$(cd texmf && pwd)"
for _ in 1 2; do
    pdflatex -interaction=nonstopmode "\pdfmapfile{+carlito.map}\input{main}" > build.log 2>&1 || true
done
grep -E "^! " build.log | sort | uniq -c || true
mv main.pdf Final_Presentation.pdf
rm -f main.aux main.log main.nav main.out main.snm main.toc main.vrb build.log
pdfinfo Final_Presentation.pdf | grep Pages
