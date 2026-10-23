# Paper manuscript

## ⚠️ Which version was submitted to PRL?

**Submitted to Physical Review Letters on 2026-07-28** (git tag `PRL-SUBMITTED-2026-07-28`).
The `PRL_SUBMITTED_2026-07-28_*` files are frozen, byte-identical copies of what was submitted.
**Never edit them.**

| File | What it is |
|---|---|
| `PRL_SUBMITTED_2026-07-28_main_letter_with_endmatter.tex` / `.pdf` | **SUBMITTED Letter** (two-column, with End Matter). Copy of `main_twocolumn.tex`/`.pdf` as of 2026-07-28 20:32 |
| `PRL_SUBMITTED_2026-07-28_supplemental_material.tex` / `.pdf` | **SUBMITTED Supplemental Material**. Copy of `supplementary.tex`/`.pdf` as of 2026-07-28 19:25 |
| `../cover_letter/PRL_SUBMITTED_2026-07-28_cover_letter.pdf` | **SUBMITTED cover letter** (last edited 2026-07-23) |

The other files are **not** the submitted version:

| File | What it is |
|---|---|
| `DRAFT_2026-07-23_NOT_SUBMITTED_*` | July 23 draft without End Matter. It was previously misnamed `main_twocolumn_SUBMITTED.tex`. **Not submitted.** |
| `DIFF_draft-2026-07-23_vs_SUBMITTED-2026-07-28_main_letter.*` | latexdiff from the July 23 draft to the submitted Letter |
| `main.tex` / `main.pdf` | Single-column working draft with line numbers and **no End Matter**. It differs from the submitted Letter. |
| `main_letteronly.tex` | Letter body without End Matter (used for the word count) |
| `supplementary_endmatter.tex` | Working file for the End Matter split |
| `*_natcomp_backup.tex` | Earlier Nature Comp. Sci. article-class versions |

Note: `.gitignore` excludes `*_twocolumn.tex` and `*_twocolumn.pdf`, so the working file `main_twocolumn.tex` is not in git.
The frozen `PRL_SUBMITTED_*` copies are the record.

Building the frozen Letter needs `supplementary.aux` for cross-references, so build the Supplemental Material first.

---

Local mirror of the Overleaf project — manually synced.

Overleaf project: https://www.overleaf.com/project/6945b5e18f3d3f9a572a9aaa

## Workflow

**To bring Overleaf changes into the repo:**
1. Overleaf → Menu → Source → **Download as Zip**.
2. Unzip into this folder, overwriting existing files.
3. From the repo root:
   ```bash
   git add paper/manuscript
   git commit -m "Sync manuscript from Overleaf"
   git push
   ```
4. The GitHub Action ([.github/workflows/build-paper.yml](../../.github/workflows/build-paper.yml)) will compile the PDF and attach it to the `latest-pdf` release.

**To push local edits back to Overleaf:**
Either edit on Overleaf directly, or upload modified files via Overleaf's editor (drag-and-drop into the file tree).

## Stable PDF URL

Once the first build succeeds, the compiled PDF lives at:
```
https://github.com/sososquirrel/troposwhere/releases/download/latest-pdf/<filename>.pdf
```

The `<filename>` matches whatever LaTeX produces — usually the same as your root `.tex` (e.g. `main.tex` → `main.pdf`).

## Configuration

If your root `.tex` file isn't `main.tex`, update `MAIN_TEX` in [.github/workflows/build-paper.yml](../../.github/workflows/build-paper.yml).
