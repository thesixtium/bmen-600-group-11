# Final presentation

Slides for the final course presentation, using the results of benchmark run `20261113_124400`
(`rssns_for_mi_EEG/results/benchmark/20261113_124400/`).

| Path | Content |
|---|---|
| `Final_Presentation.pdf` | The compiled deck (19 slides; slides 15 to 19 are backup slides after "Thank you!") |
| `main.tex` | The deck (Beamer) |
| `UofC_dynamic.sty`, `assets/` | University of Calgary slide template |
| `Images/` | Pictures used by the deck |
| `texmf/` | LaTeX packages the template needs that a standard TeX Live may not have: `carlito`, `fontaxes`, `mweights`, `environ`, `trimspaces` (from CTAN) |
| `build.sh` | Compiles `main.tex` into `Final_Presentation.pdf` |

**To compile:** `./build.sh` in this folder (needs `pdflatex`). One warning, "There's no line here to end", comes
from the author line of the title page and does not affect the output.
