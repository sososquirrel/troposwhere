# Response to reviewer comments (Koichi)

Working document — status legend:
`[✓]` done · `[~]` partial · `[ ]` pending · `[?]` author-decision needed · `[→]` text-only (Overleaf) · `[opt]` reviewer suggestion, optional

---

## TL;DR status

### Must-do (direct response to specific reviewer critique) — all done or near-done

| Item | Status | Notes |
|---|---|---|
| Cluster relabel 0–6 → 1–7 throughout | `[✓]` | applied in figures + main.tex + supplementary.tex |
| 3D PCA cross-ref to SI explicit | `[✓]` | `Fig.~\ref{fig:pca_3d}` resolves via xr-hyper |
| §3.1 opening rewrite (lead with PCA outcome) | `[✓]` | §3.1, lines 161–167 of main.tex |
| Clarify K=7 partition vs latent coords | `[✓]` | same paragraph |
| Cluster 2 in eigenmode 2 negative phase | `[✓]` | §3.2, line 276 |
| Fig 7 (Markov matrix) PCA axis labels | `[✓]` | added (intensity) / (organization) |
| Trajectory time series in main text | `[✓]` | new Fig 8 (`fig8_trajectories.pdf`) |
| §3.3 "Expected behavior" trimmed → SI | `[✓]` | derivation kept in SI |
| New §3.3 paragraph + 6 references | `[✓]` | Zhang, Bechtold, Piriou, Gerard, Mapes & Neale, Hwong |
| Bootstrap CI bands on profile figures | `[✓]` | Figs 5 & 6 |
| Cold-pool autocorrelation reformulation | `[✓]` | windowed (≥20 steps) + SI diagnostic |
| Eigenvalue → physical time scales | `[✓]` | computed; need to weave into text + table |
| Time-step sensitivity of τ_k (SI) | `[✓]` | new `tau_vs_dt.pdf` |
| State summary table | `[~]` | numerical columns from notebook; qualitative needs author input |

### Optional / author-decision

| Item | Status | Note |
|---|---|---|
| PCA1/PCA2 orthogonality discussion (~3 sentences in §3.1) | `[?]` `[opt]` | Reviewer raised the question; not strictly required but good to address. |
| Methods Fig 2 caligraphic notation cleanup | `[?]` | TikZ schematic outside notebook scope — need to inspect Overleaf source. |
| Cite the 6 §3.3 refs also in Introduction | `[?]` `[opt]` | Reviewer suggested cross-linking; currently only cited in §3.3. |
| References for **math tools** (autocorrelation, Markov basics, indicator functions) | `[?]` `[opt]` | Reviewer asked for textbook/review-level pointers — could add one to each. |
| Move trajectory figure (Fig 8) to SI | `[?]` | Reviewer wanted it in main text. You hinted you'd like it in SI. Easy switch. |
| Cluster 2-line qualitative descriptions in table | `[ ]` | Needs author judgement per state (e.g., "Cluster 4 = weakly organized intense convection"). |
| Move §3.1 to make more readable (overall reviewer note) | `[opt]` | The "main text too technical" general feedback — already partly addressed by §3.3 trim. |

### Done and verified building

- Local build: 22-page main.pdf + 12-page supplementary.pdf, zero unresolved references.
- CI: `.github/workflows/build-paper.yml` now builds SI first then main → `xr-hyper` cross-refs resolve.

**You don't have to do everything in the optional section.** Pick what you want to actually address; the rest can be left as "the reviewer's suggestion is noted but we feel X is sufficient" in the response letter.

---

## Overall impression

> *Main text is too technical; move technical material to SI; add more in-depth scientific discussion and references in the Results section.*

`[→]` **Acknowledged.** Concretely we will:
- Move the "Expected behavior for a Markov chain" subsection (§3.3) to the SI, keeping only Eqs. 14–15 and the relaxation-time argument in the main text — see point §3.3 below.
- Add references for indicator functions, normalized autocorrelation, and Markov-chain basics (textbook/review-level pointers) in §3.3 and Methods.
- Cite the convection-memory / quasi-equilibrium literature suggested by the reviewer in §3.3 and the Introduction (Bechtold 2014, Hwong 2023, Gerard 2009, Mapes & Neale 2011, Piriou 2007, Zhang 2003).
- Cross-link Introduction citations into the Results section where appropriate.

## 1. Introduction
> *Reads fine.*

No action.

## 2. Methods — Figure 2 (reduced-order stochastic model schematic)

> *Caligraphic notations in the figure don't clearly correspond to the symbols in the caption and main text.*

`[→]` TikZ figure in the Overleaf source; will go through the schematic and the captions and force notation to match the equations in §2.3 line-for-line. Action item for Sophie when revising the Methods section.

## 3. Results

### 3.1 Physical interpretation of the latent space

#### Opening of §3.1 (motivation for PCA)
> *Lead with the outcome of PCA up front — e.g., "PCA separates the seven clusters into a 2D phase space, with one axis for intensity and one for organization."*

`[→]` We will rewrite the first paragraph and first half of the second so the section opens with the PCA result, then explains the mechanics. Draft text to follow in a separate text-revisions block.

#### Clarify "individual latent coordinates" vs K=7
> *Confusing alongside §2.3.2 ("the continuous latent representation is mapped onto a finite set of K = 7 discrete convective states"), which implies the K=7 partition already provides physical meaning.*

`[→]` We will clarify the distinction: the K=7 partition captures *which* convective regime, but the **ordering** of the seven states is arbitrary by construction (the VAE places them in latent space without imposing intensity/organization axes). PCA is what reveals an interpretable 2D ordering of those clusters. Draft text to follow.

#### Figure 3 — state summary table
> *Add a table or bullet list with each state's organization regime, time scale, etc.*

`[~]` Computing the per-state quantitative entries now (stationary probability, exponential relaxation time from leading eigenmodes). Qualitative organization label for each state needs Sophie's input — once the numerical fields are filled in we'll add the column-by-column descriptions together.

#### PCA1 / PCA2 orthogonality
> *Reviewers may ask why intensity and organization appear as orthogonal axes, given they can be highly correlated in real-world MCSs.*

`[→]` Worth addressing explicitly in the text. PCA orthogonalizes by construction (PCA2 is the residual after PCA1), so the axes can appear decorrelated in the latent embedding even when the underlying physical quantities are correlated in observations. We will add 2–3 sentences making this caveat explicit, and note that the observed orthogonality is partly a consequence of the VAE training on idealized RCE-like simulations where intensity and organization can co-vary along multiple distinct trajectories.

#### Cluster numbering convention
> *0-based indexing seems unnatural in a paper.*

`[~]` Relabeling all figures and the manuscript to 1-indexed states (1–7). Implementation: a `state_label()` helper in `workflow_paper.py` shifts the display label by +1 so figure legends/annotations show 1–7; the underlying data array stays 0-indexed (no model/code changes needed). Figure regeneration is required for clusters / vertical_profiles / mse_profiles / latent_axis / markov_matrix / markovianity_and_coldpools and the SI panels that label states. Manuscript text also needs every "State 0" → "State 1", "Cluster 6" → "Cluster 7", etc.

#### Missing 3D PCA reference
> *"A three-dimensional PCA projection, shown in the Supplementary Material, confirms that clusters with apparent overlap in two dimensions are separated along higher-order components." — I cannot find a corresponding material in the supplementary.*

`[✓ figure exists]` / `[→ ref to fix]` The 3-component PCA panel is already produced as `pca_3d.pdf` (SI Fig. 7, from cell 18 of [figures_si.ipynb](paper/notebooks/figures_si.ipynb)). The manuscript pointer is too vague — fix to an explicit cross-reference (e.g., `\ref{fig:pca_3d}` or "Fig. SI 7").

#### Figures 5 and 6 — confidence intervals
> *Add visualization of uncertainty/confidence to identify the most significant features.*

`[ ]` To add: per-state mean profile ± bootstrap (or s.e.m.) shaded band on Figs. 2 and 3 (notebook) → Figs. 5 and 6 in the paper. Cells [c458c6f6](paper/notebooks/figures_paper.ipynb) (`vertical_profiles.pdf`) and [36796f9d](paper/notebooks/figures_paper.ipynb) (`mse_profiles.pdf`).

### 3.2 Markov transition matrix, eigenmodes, and regimes

#### Cluster 2 in eigenmode 2 negative phase
> *"In the negative phase, Clusters 3 and 5 are enhanced at the expense of Cluster 6 ... How about cluster 2?"*

`[ ]` Will compute and report the amplitude of eigenmode 2 on cluster 2 (whether it's noise-level <0.1 or comparable to clusters 3, 5, 6) and update the text accordingly.

#### Figure 7 — axis labels
> *Add "(intensity)" and "(organization)" to PCA1 and PCA2 axis labels.*

`[✓]` Done in cell [327eb91d](paper/notebooks/figures_paper.ipynb); also applied for consistency to the clusters figure (Fig. 3) and the latent_axis figure (Fig. 4).

#### Time-series of trajectories in main text
> *"Time series of individual realizations confirm that system trajectories cluster along these two pathways..." — please include such figures in the main text.*

`[ ]` Will add a new main-text figure showing either (a) horizontal snapshots of mass flux along two representative trajectories (one for each pathway), or (b) domain-wide time series of PW / precip / W500 colored by state for two realizations — or both. Sophie to decide on layout.

### 3.3 Markovianity assessment and internal feedbacks

#### References on memory of convective systems
`[→]` Will add the six references suggested (Bechtold 2014, Hwong 2023, Gerard 2009, Mapes & Neale 2011, Piriou 2007, Zhang 2003) to §3.3 and the Introduction. Specifically:
- Bechtold 2014 → §3.3 intro paragraph (equilibrium vs nonequilibrium convection).
- Hwong 2023 → §3.3 next to the autocorrelation analysis (memory in convection schemes).
- Gerard 2009, Piriou 2007 → Methods (parameterizations with memory).
- Mapes & Neale 2011 → Introduction (convective organization parameterization).
- Zhang 2003 → §3.3 (quasi-equilibrium in tropical western Pacific).

#### Sensitivity of memory analysis to data time step
> *The memory is tied to the time step of the model — present sensitivity to data time step, and/or explicit values of exponential time scales from each eigenvalue.*

`[~]` We will:
1. Compute the exponential relaxation time `τ_k = -Δt / log(|λ_k|)` for each eigenvalue of the transition matrix and report them in physical units (minutes/hours). These will be added to the state summary table requested for Fig. 3.
2. Refit the transition matrix on subsampled trajectories (e.g., Δt × 2, 5, 10) and show that the leading eigenvalues / relaxation times are consistent across resolutions. New SI panel.

Note: cell [cfe4dac0](paper/notebooks/figures_paper.ipynb) already does a partial version of this (comparing empirical `T(k)` to `exp((k/5)·log(T(5)))`) — we will repackage for the SI.

#### "Expected behavior for a Markov chain" subsection → SI
> *Move to SI, keep Eq. 14, 15 and last two paragraphs in main text.*

`[→]` Confirmed plan: keep equations 14 and 15 + the two closing paragraphs that link eigenvalues to exponential timescales in the main text; move the derivation/exposition (autocorrelation under a Markov chain, etc.) to the SI with a clear reference.

#### Figure 8 — cold-pool autocorrelation reformulation
> *"Conditioning on cold pool activity" needs more clarification. If cold pool conditioning makes the indicator time series discontinuous, then it artificially reduces autocorrelation. Instead, require cold pool conditions to continue for >20 or 30 time steps, calculate autocorrelation per cold pool sample, then average.*

`[ ]` **Methodological priority.** Reimplementation plan:
1. Identify all contiguous time windows where cold-pool intensity stays in the "moderate" / "strong" bin for at least N consecutive time steps (try N = 20 and N = 30 as a sensitivity check).
2. For each such window, compute the state-6 indicator autocorrelation up to a lag of min(window length, max_lag).
3. Average autocorrelations across windows (weighted by window length or unweighted — note both).
4. Replace the current chopped-series version in cell [1ce52342](paper/notebooks/figures_paper.ipynb) and the function `indicator_autocorrelation_state6_cp` in [workflow_paper.py](paper/workflow_paper.py).
5. Add a short SI panel showing an example cold-pool-intensity time series and the corresponding state-6 indicator for illustration.

---

## References to add (reviewer-suggested)

The §3.3 citation paragraph in [main.tex](manuscript/main.tex#L304) cites these in two groups:
- *Quasi-equilibrium / equilibrium adjustment*: `\citep{zhang2003convective,bechtold2014representing}`
- *Memory-enabled parameterizations*: `\citep{piriou2007approach,gerard2009cloud,mapes2011parameterizing}`
- *Memory testing of schemes*: `\citep{hwong2023assessing}`

### Bibliography notes — what each paper contributes

**Zhang 2003** — *Convective quasi-equilibrium in the tropical western Pacific.* J. Geophys. Res., 108(D19).
- Tests whether convective intensity is in equilibrium with large-scale CAPE generation by comparing CAPE evolution against expected adjustment timescales in TWP-ICE soundings.
- **Conclusion we cite**: provides the canonical observational support for the *quasi-equilibrium ansatz* — convection adjusts rapidly relative to forcing, so a memoryless mapping from forcing to convection is defensible in tropical maritime regimes.
- **Why it matters for us**: our Markov ansatz is essentially a *discrete-state* quasi-equilibrium hypothesis at a 10-min step. Zhang sets the precedent that this assumption is workable for some regimes but not universal.

**Bechtold et al. 2014** — *Representing equilibrium and nonequilibrium convection in large-scale models.* J. Atmos. Sci., 71(2), 734–753.
- Modifies the ECMWF IFS convection scheme to separately handle "equilibrium" (tropical, fast-adjusting) and "nonequilibrium" (mid-latitude, lagged) convection by extending the closure timescale.
- **Conclusion we cite**: explicit demonstration that *one closure timescale is insufficient* across regimes — non-equilibrium convection requires longer effective adjustment / memory.
- **Why it matters for us**: directly motivates the existence of internal-feedback memory (cold pools, mesoscale circulations) that *causes* deviations from Markovianity. Bechtold's "non-equilibrium correction" is essentially what our autocorrelation analysis detects.

**Piriou et al. 2007** — *An Approach for Convective Parameterization with Memory.* J. Atmos. Sci., 64(11), 4127–4139.
- Introduces prognostic variables for cloud water and precipitation in the convection scheme of Météo-France's ARPEGE, separating microphysics from transport.
- **Conclusion we cite**: pioneering example of *explicit memory variables* inside a convection parameterization — convection at time $t+\Delta t$ depends not only on environment but also on remnant cloud water from earlier.
- **Why it matters for us**: prior art for treating convection as a *non-memoryless* process. Provides the conceptual template for adding cold-pool state as a memory variable (what we hypothesise restores Markovianity in our framework).

**Gerard et al. 2009** — *Cloud and Precipitation Parameterization in a Meso-Gamma-Scale Operational Weather Prediction Model.* Mon. Wea. Rev., 137(11), 3960–3977.
- Builds the ALARO scheme on top of Piriou's framework, with explicit prognostic memory of convective updraft mass flux and entrainment, designed for meso-γ resolutions where the convective grey zone exposes scheme memory problems.
- **Conclusion we cite**: operational follow-up to Piriou 2007; demonstrates that prognostic memory variables are *necessary* at resolutions approaching the convective scale.
- **Why it matters for us**: our simulations are at ~1 km — exactly the grey-zone resolution that Gerard et al. argue requires memory.

**Mapes & Neale 2011** — *Parameterizing convective organization to escape the entrainment dilemma.* JAMES, 3(2).
- Adds a prognostic "organization" variable to a CAM convection scheme that lags the convective response — when organization is high, entrainment is reduced and deep convection is favoured.
- **Conclusion we cite**: convective *organization itself* is a slow internal variable that needs to be tracked explicitly. Memory ≠ moisture memory only; geometric/dynamical organization is also memory.
- **Why it matters for us**: this is the closest spiritual precursor to our work. They prescribe organization as a memory variable; we *learn* an organization axis (PCA 2) and *measure* the memory it implies.

**Hwong et al. 2023** — *Assessing Memory in Convection Schemes Using Idealized Tests.* JAMES, 15(12), e2023MS003726.
- Designs an idealized switch-off forcing test to measure how long different parameterizations "remember" prior convective activity after the forcing is removed, comparing CAM, IFS, and others against CRM truth.
- **Conclusion we cite**: provides a *quantitative methodology* for measuring memory in convection schemes — many schemes have effectively zero memory; CRMs have substantial memory.
- **Why it matters for us**: validates the value of *measuring* memory (as we do with autocorrelation analysis) rather than assuming it. Their framework and our framework are complementary — they test parameterizations against CRMs; we use CRMs to identify *where* memory lives.

### Full references

- Bechtold, P., Semane, N., Lopez, P., Chaboureau, J.-P., Beljaars, A., & Bormann, N. (2014). *Representing equilibrium and nonequilibrium convection in large-scale models.* J. Atmos. Sci., 71(2), 734–753. https://doi.org/10.1175/JAS-D-13-0163.1
- Hwong, Y.-L., Colin, M., Aglas-Leitner, P., Muller, C. J., & Sherwood, S. C. (2023). *Assessing Memory in Convection Schemes Using Idealized Tests.* JAMES, 15(12), e2023MS003726. https://doi.org/10.1029/2023MS003726
- Gerard, L., Piriou, J.-M., Brožková, R., Geleyn, J.-F., & Banciu, D. (2009). *Cloud and Precipitation Parameterization in a Meso-Gamma-Scale Operational Weather Prediction Model.* Mon. Wea. Rev., 137(11), 3960–3977. https://doi.org/10.1175/2009MWR2750.1
- Mapes, B., & Neale, R. (2011). *Parameterizing convective organization to escape the entrainment dilemma.* JAMES, 3(2). https://doi.org/10.1029/2011MS000042
- Piriou, J.-M., Redelsperger, J.-L., Geleyn, J.-F., Lafore, J.-P., & Guichard, F. (2007). *An Approach for Convective Parameterization with Memory: Separating Microphysics and Transport in Grid-Scale Equations.* J. Atmos. Sci., 64(11), 4127–4139. https://doi.org/10.1175/2007JAS2144.1
- Zhang, G. J. (2003). *Convective quasi-equilibrium in the tropical western Pacific: Comparison with midlatitude continental environment.* JGR, 108(D19). https://doi.org/10.1029/2003JD003520
