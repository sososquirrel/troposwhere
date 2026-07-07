# Response to Reviewers

Thank you both for the careful and constructive reading. Below I reproduce each
of your comments (in **) and answer them in turn. Everything has been
incorporated into the revised manuscript and Supplementary Material; figure and
section numbers refer to the revised version.

---

# Koichi

## Overall

> *The main text is too technical; technical material should move to the SI,
> with more in-depth scientific discussion and references in the Results.*

I agree, and I've restructured accordingly. I moved the "Expected behavior for a
Markov chain" exposition (former §3.3) to the Supplementary Material and kept
only Eqs. 14–15 and the relaxation-time argument in the main text. I added
review-level references for indicator functions, normalized autocorrelation, and
Markov-chain basics in §3.3 and the Methods, and I now cite the
convection-memory / quasi-equilibrium literature you pointed me to (Zhang 2003,
Bechtold 2014, Piriou 2007, Gerard 2009, Mapes & Neale 2011, Hwong 2023) in both
§3.3 and the Introduction, with the relevant citations cross-linked into the
Results.

## 1. Introduction

> *Reads fine.*

Thanks, but I have changed a bit the phrasing since then :-)

## 2. Methods — Figure 2

> *Calligraphic notation in the figure does not clearly correspond to the
> symbols in the caption and main text.*

You're right. I revised the schematic and its caption so the notation now
matches the symbols in the §2.3 equations line for line.

## 3. Results

### 3.1 Latent space

> *Lead with the outcome of the PCA up front (e.g., "PCA separates the seven
> clusters into a 2D phase space, one axis for intensity, one for organization").*

Good suggestion — I rewrote the opening of §3.1 to state the PCA result first
and then explain the mechanics.

> *"Individual latent coordinates" is confusing alongside §2.3.2, which implies
> the K = 7 partition already provides physical meaning.*

I see the confusion and have clarified it: the K = 7 partition captures *which*
convective regime a snapshot belongs to, but the *ordering* of the seven states
is arbitrary by construction — the VAE places the clusters in latent space
without imposing intensity or organization axes. The PCA is what reveals an
interpretable 2D ordering of those clusters.

> *Add a table or bullet list with each state's organization regime, time
> scale, etc.*

Done — I added a state summary table to Fig. 3 giving, per state, the stationary
probability π, the exponential relaxation time from the leading eigenmodes (in
physical units), and a one-line qualitative descriptor of its organization
regime.

> *Why do intensity and organization appear as orthogonal axes, given they can
> be highly correlated in real MCSs?*

This is worth being explicit about, and I now address it in §3.1. PCA
orthogonalizes by construction (PCA2 is the residual after PCA1), so the two
axes can appear decorrelated in the latent embedding even when the underlying
physical quantities are correlated in observations; their physical drivers
(cold-pool forcing, shear-relative updraft tilt) remain coupled along
trajectories. I added a couple of sentences making this caveat clear.

> *0-based cluster indexing is unnatural in a paper.*

Agreed — all figures and text now use 1-indexed states (1–7).

> *I cannot find the three-dimensional PCA projection referenced for the SI.*

Sorry about that — the 3-component PCA panel is in the Supplementary Material,
and I've replaced the vague pointer in the main text with an explicit
cross-reference to that figure.

> *Add a visualization of uncertainty/confidence on Figs. 5 and 6.*

Done — Figs. 5 and 6 now show per-state mean profiles with bootstrap confidence
bands.

### 3.2 Markov matrix and eigenmodes

> *In the negative phase, Clusters 3 and 5 are enhanced at the expense of
> Cluster 6 … how about Cluster 2?*

I computed the eigenmode-2 amplitude on Cluster 2 and now report it in the text
alongside the other clusters.

> *Add "(intensity)" and "(organization)" to the PCA1 and PCA2 axis labels.*

Done, and applied consistently to Figs. 3, 4, and 7.

> *Please include the time series of individual trajectories in the main text.*

I added a new main-text figure (Fig. 8) showing domain-wide time series along
two representative trajectories, one for each preferred pathway.

### 3.3 Markovianity and feedbacks

> *Add references on the memory of convective systems.*

I added and integrated them as follows: Bechtold 2014 (equilibrium vs.
nonequilibrium convection) and Zhang 2003 (quasi-equilibrium in the tropical
western Pacific) in the §3.3 introduction; Hwong 2023 next to the
autocorrelation analysis; Piriou 2007 and Gerard 2009 in the Methods
(memory-enabled parameterizations); Mapes & Neale 2011 in the Introduction. A
short rationale for each is in the bibliography note below.

> *Memory is tied to the model time step — show sensitivity to the data time
> step, and/or explicit exponential time scales from each eigenvalue.*

I did both. I compute the exponential relaxation time τ_k = −Δt / log|λ_k| for
each eigenvalue and report it in physical units in the Fig. 3 state table. I
also refit the transition matrix on subsampled trajectories (Δt × 2, 5, 10) and
show in a new SI panel that the leading eigenvalues and relaxation times are
consistent across resolutions.

> *Move the "Expected behavior for a Markov chain" subsection to the SI, keeping
> Eqs. 14, 15 and the last two paragraphs in the main text.*

Done exactly as you suggested: Eqs. 14–15 and the two closing paragraphs linking
eigenvalues to exponential timescales stay in the main text; the derivation
moves to the SI with a cross-reference.

> *"Conditioning on cold pool activity" needs clarification — discontinuous
> conditioning artificially reduces autocorrelation. Instead require cold-pool
> conditions to persist for >20–30 steps, compute autocorrelation per sample,
> then average.*

You're right that the discontinuity was a problem, and I reimplemented the
analysis the way you proposed. I now identify contiguous windows where cold-pool
intensity stays in the moderate/strong bin for at least N consecutive steps
(N = 20 and N = 30, as a sensitivity check), compute the state-7 indicator
autocorrelation within each window, and average across windows (I report both
length-weighted and unweighted means). A new SI panel shows an example
cold-pool-intensity series with the corresponding state-7 indicator for
illustration.

## Bibliography note (on the references you suggested)

The §3.3 citation paragraph cites these in three groups — quasi-equilibrium
(Zhang 2003, Bechtold 2014), memory-enabled parameterizations (Piriou 2007,
Gerard 2009, Mapes & Neale 2011), and memory testing of schemes (Hwong 2023):

- **Zhang 2003** — Observational support for the quasi-equilibrium ansatz;
  convection adjusts rapidly relative to forcing in tropical maritime regimes.
  Sets the precedent that my discrete-state quasi-equilibrium hypothesis is
  workable for some regimes but not universal.
- **Bechtold et al. 2014** — Shows one closure timescale is insufficient across
  regimes; non-equilibrium convection needs longer effective memory. Motivates
  the internal feedbacks (cold pools) that cause deviations from Markovianity.
- **Piriou et al. 2007** — Pioneering explicit memory variables in a convection
  scheme; the conceptual template for adding cold-pool state as a memory
  variable.
- **Gerard et al. 2009** — Operational follow-up to Piriou; prognostic memory is
  necessary at grey-zone resolutions (~1 km), exactly my regime.
- **Mapes & Neale 2011** — Treats convective organization itself as a slow
  internal memory variable; the closest precursor — they prescribe
  organization, I learn it (PCA2) and measure the memory it implies.
- **Hwong et al. 2023** — Quantitative methodology for measuring scheme memory;
  validates *measuring* memory rather than assuming it, complementary to my
  autocorrelation analysis.

Full references:

- Bechtold, P., et al. (2014). Representing equilibrium and nonequilibrium
  convection in large-scale models. *J. Atmos. Sci.*, 71(2), 734–753.
- Gerard, L., et al. (2009). Cloud and Precipitation Parameterization in a
  Meso-Gamma-Scale Operational Weather Prediction Model. *Mon. Wea. Rev.*,
  137(11), 3960–3977.
- Hwong, Y.-L., et al. (2023). Assessing Memory in Convection Schemes Using
  Idealized Tests. *JAMES*, 15(12), e2023MS003726.
- Mapes, B., & Neale, R. (2011). Parameterizing convective organization to
  escape the entrainment dilemma. *JAMES*, 3(2).
- Piriou, J.-M., et al. (2007). An Approach for Convective Parameterization
  with Memory. *J. Atmos. Sci.*, 64(11), 4127–4139.
- Zhang, G. J. (2003). Convective quasi-equilibrium in the tropical western
  Pacific. *JGR*, 108(D19).

---

# Huancui

**Note on state labels.** Your comments use the 0-indexed labels (states 0–6)
of the version you reviewed. The revised manuscript uses 1-indexed labels
(states 1–7); below I answer with the new labels and give your original label in
parentheses. Mapping:

| Your label (0-idx) | Revised (1-idx) | Role | π |
|---|---|---|---|
| 0 | 1 | rare bridge state | 4.6% |
| 1 | 2 | bridge between regimes | 3.5% |
| 2 | 3 | suppressed / quiescent | 36.4% |
| 3 | 4 | intense, weakly-organized cells | 19.1% |
| 4 | 5 | rare in-between state | 4.0% |
| 5 | 6 | popcorn / weakly-organized | 19.7% |
| 6 | 7 | sustained deep, strongly-organized | 12.7% |

## Comment 1 — Why is the most organized pathway not the most intense?

> *PCA1 indicates precipitation intensity and PCA2 indicates organization. The
> strongly organized trajectory oscillating between clusters 2 and 6 [States 3
> and 7] is not necessarily the most intense; with 2↔6 a preferred pathway (the
> other being 2-5-3 [State 3 ↔ 6 ↔ 4]), the most organized states are not often
> the most intense, which is counter-intuitive. How would you interpret that?*

Your intuition is actually correct once "intensity" is read as the peak local
precipitation rate rather than the domain mean. **State 7, the most organized
state, unambiguously produces the most extreme precipitation** — both in the
frequency of extreme events and in the absolute peak rate.

The methodological point: domain-mean precipitation, which I previously used to
motivate PCA1 as an "intensity" axis, *dilutes* organized convection, because
squall-line cores cover only a fraction of the 128×128 km domain. The
appropriate metric is the **spatial-max precipitation**. With global thresholds
Q₉₅ = 2720 and Q₉₉ = 3547 (relative units of the 2D Prec field):

| State (rev.) | Mean spatial-max | Q₉₉ | Max | P(P_max > Q₉₉ \| s) | Share of >Q₉₉ events | π |
|---|---|---|---|---|---|---|
| 3 (old 2) | 1095 | 2804 | 5487 | 0.2% | 6.7% | 36.4% |
| 4 (old 3) | 2020 | 3866 | 5865 | 1.6% | 30.8% | 19.1% |
| 6 (old 5) | 1381 | 2532 | 3975 | 0.05% | 1.0% | 19.7% |
| 7 (old 6) | **2011** | **4469** | **6510** | **3.6%** | **46.2%** | 12.7% |

State 7 produces the highest spatial-max rate ever observed in the 19,476-step
simulation and the heaviest Q₉₉ tail; it is more than twice as likely as State 4
to produce a >Q₉₉ event per unit time; and it captures 46% of all extreme events
while occupying only 12.7% of the time. The least-organized active state
(State 6, popcorn) captures 1% of extreme events despite 19.7% of the time — a
~30× gap relative to its time share.

So your intuition holds, provided "intensity" means peak local precipitation.
The mismatch came from the domain mean, which collapses an organized squall line
(intense core over ~30% of the domain) and disorganized vigorous convection
(moderate cells over the whole domain) to comparable numbers, even though the
two regimes differ greatly in physics and in flood-relevant local rate.
Physically, organized convection (State 7) concentrates buoyancy through the
cold-pool–shear feedback into deep updrafts with a narrow, high-intensity core,
whereas the popcorn state lacks sustained cold-pool dynamics and its cells stay
shallow and short-lived. PCA1 and PCA2 are near-orthogonal in the PCA basis (by
construction) but their physical drivers are coupled across trajectories.


## Comment 2 — Why is the State 5 → State 3 transition not in Fig. 7a?

> *The value from state 4 to state 2 [State 5 → State 3] is 0.16, high compared
> with other transitions. Does this mean an initial state of 4 very likely
> transits to state 2? If so, why is this link not in Figure 7a — because the
> transition is not particularly enhanced or suppressed in the dominant mode?*

Your hypothesis is essentially correct; two factors combine.

First, State 5 (old 4) is rarely occupied (π₅ ≈ 4.0%). The Fig. 7 caption states
that transitions involving States 1 and 5 are omitted from panel (a) for visual
clarity; they remain in panel (b), which is why you can still see the 0.16
entry.

Second, State 5 carries an amplitude of only −3% in the leading non-trivial
eigenvector **v₂** = [-0.11, -0.19, +0.30, -0.49, -0.02, -0.44, +0.65] — effectively
zero. So State 5 is not a participant in the dominant slow mode that organizes
the chain into the two pathways of §3.2; omitting it from Fig. 7a hides no
feature of the dominant dynamics.

This is sharper in the **unconditional flux** π₅·T₅ⱼ than in the conditional
T₅ⱼ. The largest outgoing transition, 5 → 3 (T = 0.163), contributes only
~0.65% of all transitions globally, whereas π₄·T₄₆ (State 4 → 6, a
within-weak-regime transition) is ~1.5% — more than twice as large despite a
*smaller* conditional probability (T₄₆ = 0.076). The arrow diagram is drawn from
the flux, not from the matrix row: your reading of the matrix is correct, but
the matrix value alone overstates this transition's dynamical importance.

I'll keep the figure as is and extend the Fig. 7 caption to make the criterion
explicit: arrows in panel (a) are proportional to the unconditional flux πᵢTᵢⱼ
rather than Tᵢⱼ alone, and transitions involving States 1 and 5 are omitted
because π₁, π₅ < 5% and both carry |**v₂**| < 0.2, contributing negligibly to
the dominant mode.

## Comment 3 — What is the ultimate purpose of this framework?

> *It is not clear what the ultimate purpose of this framework is. Is it mainly
> to understand MCS processes, or could it build a simplified model to predict
> convective systems? Maybe expand a little on how to use it.*

A fair point — the manuscript currently emphasises the diagnostic side and
underplays the prospective uses. I change the conclusion part in this sense and signal these in the abstract and
introduction:

1. **Process understanding (this paper).** A dynamically meaningful coordinate
   system for convective variability: PCA1/PCA2 separate lifecycles by intensity
   and organization, the Markov chain identifies preferred state sequences, and
   residual non-Markovianity diagnoses internal feedbacks (cold pools here, but
   the methodology generalises). This lets me *test* memory hypotheses directly
   from CRM output in a low-dimensional space where statistical questions have
   well-defined answers.

2. **A learned stochastic parameterization.** The transition matrix **T** over
   the seven states is itself a candidate simplified model; because the latent
   representation is decoded by the VAE to a mass-flux profile, sampling the
   chain yields a generative model of convective mass flux consistent with the
   CRM statistics — a data-driven alternative to prescribed multi-cloud Markov
   schemes (Khouider et al. 2010). Adding cold-pool intensity to the state
   description (§3.3) is a constructive recipe for improving this emulator.

3. **Diagnosing memory in existing parameterizations.** The autocorrelation /
   survival diagnostics (Fig. 8) are transferable the seven-state Markov reference is a quantitative null against which
   a scheme's memory structure can be compared. A scheme lacking cold-pool
   memory will fail to reproduce the survival-curve heterogeneity of Fig. 8b; a
   scheme with too much memory will overshoot it.

I'll reframe the closing paragraphs of §4 to present the framework
prospectively as a tool for building and testing stochastic convective
representations, not only as a diagnostic of the current simulations.
