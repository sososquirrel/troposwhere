# Paper manuscript

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
