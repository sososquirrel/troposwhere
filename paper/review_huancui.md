# Response to reviewer comments (Huan Cui)

Working document — status legend:
`[✓]` done · `[~]` partial · `[ ]` pending · `[?]` author-decision needed · `[→]` text-only

---

## Labeling note

Huan Cui's comments refer to **0-indexed** state labels (states 0–6), as in the version of the manuscript he reviewed. The current manuscript uses **1-indexed** labels (states 1–7) — see the response to Koichi ([review_koichi.md](review_koichi.md), §3.1). For convenience, the mapping is:

| Reviewer (0-indexed) | Current paper (1-indexed) | Cluster colour | Role / character |
|---|---|---|---|
| 0 | 1 | lavender | rare bridge state (π = 4.6%) |
| 1 | 2 | pink     | bridge between regimes (π = 3.5%) |
| 2 | 3 | yellow   | suppressed / quiescent, between bursts (π = 36.4%) |
| 3 | 4 | blue     | intense, weakly-organized cells (π = 19.1%) |
| 4 | 5 | mint     | rare in-between state (π = 4.0%) |
| 5 | 6 | green    | popcorn / weakly-organized convection (π = 19.7%) |
| 6 | 7 | red      | sustained deep, strongly-organized convection (π = 12.7%) |

Below, we quote the reviewer's wording verbatim and respond using the **new 1-indexed** labels, with the reviewer's old label in parentheses for cross-reference.

---

## Comment 1 — Why is the most organized pathway not the most intense?

> *In your results about the two dominant PCA modes, PCA1 indicates the precipitation intensity and PCA2 indicates the organization. In the trajectories shown in Figure 3, the one with strongly organized development that oscillates between clusters 2 and 6 [→ States 3 and 7 in the new labels] is not necessarily with the strongest precipitation intensity, correct? With the oscillation between 2 and 6 being one of the preferred pathway (the other being 2-5-3) [→ State 3 ↔ State 6 ↔ State 4], it seems that the most organized states are not often associated with the most intense states. This is a bit counter intuitive to me, because I would assume the most organized states would lead to the most intense precipitation. How would you interpret that?*

**Response.** The reviewer's intuition is *correct* once "intensity" is read as the peak local precipitation rate rather than the domain-mean. **State 7 (the most organized state) unambiguously produces the most extreme precipitation** — both by the frequency of extreme events and by the absolute peak rate.

The key methodological point: domain-mean precipitation, which we used previously to motivate PCA1 as an "intensity" axis, *dilutes* organized convection because squall-line cores cover only a fraction of the 128×128 km domain. The right metric for an "intensity" axis is the **spatial-max precipitation** — the peak rate anywhere in the domain at each timestep. With this metric, using global thresholds $Q_{95} = 2720$ and $Q_{99} = 3547$ (relative units of the 2D Prec field):

| State (new) | Mean of spatial-max | $Q_{99}$ of spatial-max | Max | $\mathrm{P}(P_{\max} > Q_{99}\mid s)$ | Share of all $>Q_{99}$ events | $-\mathrm{Var}_{\mathrm{spatial}}(\mathrm{PW})$ | π |
|---|---|---|---|---|---|---|---|
| 3 (old 2) | 1095 | 2804 | 5487 | 0.2% | 6.7% | -9.35 | 36.4% |
| 4 (old 3) | 2020 | 3866 | 5865 | 1.6% | 30.8% | -6.07 | 19.1% |
| 6 (old 5) | 1381 | 2532 | 3975 | 0.05% | 1.0% | **-3.55** (least org.) | 19.7% |
| 7 (old 6) | **2011** | **4469** | **6510** | **3.6%** (highest) | **46.2%** (highest) | **-16.87** (most org.) | 12.7% |

Three facts stand out:

- **State 7 produces the absolute highest spatial-max precipitation rate ever observed** in the 19,476-step simulation (6510 vs State 4's 5865), and the heaviest $Q_{99}$ tail.
- **State 7 is the most likely state to produce a $>Q_{99}$ event** per unit time: 3.6% of State 7 timesteps cross the extreme threshold, more than twice State 4's 1.6% and an order of magnitude above any other state.
- **State 7 captures 46% of all extreme events** despite occupying only 12.7% of total time. The least-organized active state (State 6, popcorn) captures 1% of extreme events despite occupying 19.7% of total time — a 30× gap relative to its time share.

So the reviewer's intuition that "the most organized states should give the most intense precipitation" is correct, provided "intensity" is interpreted as the peak local precipitation rate. The mismatch in our previous framing came from using the domain-mean, which collapses an organized squall line (intense core over ~30% of the domain) and disorganized vigorous convection (moderate cells distributed over the whole domain) to comparable numbers. The two regimes are very different in their *physics* (and in their flood-relevant local precipitation rate) even when their domain-averaged precip looks similar.

Physically:

1. **PCA1 orders states by their capacity to produce extreme local precipitation.** Organized convection (State 7) concentrates buoyant air through the cold-pool–shear feedback, producing very deep updrafts with high condensation rates in a narrow convective core. Disorganized convection (State 4) has many independent moderate cells; its domain-mean is competitive, but no single cell reaches the same peak intensity as the squall-line core.

2. **The disorganized popcorn state (State 6) cannot produce $>Q_{99}$ events at all** (only 1% of all extreme events arise there despite it occupying 20% of all timesteps). This is the same physics interpreted from the other end: popcorn convection lacks sustained cold-pool dynamics, so individual cells stay shallow and short-lived.

3. **PCA1 (intensity, in the extreme-local-precip sense) and PCA2 (organization) are physically linked.** They are nearly orthogonal in the PCA basis because PCA decorrelates by construction, but their underlying physical drivers (cold-pool forcing, shear-relative updraft tilt) are coupled. The latent representation faithfully separates them as independent *modes of variability*, but the actual values they take are correlated *across the trajectory*.

We will replace the mean-precipitation framing of PCA1 in §3.1 with a spatial-max / extreme-precipitation framing, and add a panel showing the per-state extreme-precipitation distribution alongside the existing mean. This both resolves Huan Cui's question and gives PCA1 a sharper physical meaning. `[→]`

**Supporting figures produced**:
- 3-panel summary: [figures/review_huancui/intensity_organization_and_state5_flux.pdf](figures/review_huancui/intensity_organization_and_state5_flux.pdf), panels **(a)** (per-state probability of $>Q_{95}$ and $>Q_{99}$ events) and **(b)** (extreme-event probability vs organization — State 7 alone at the bottom-right of the plane).
- Latent-space view: [figures/review_huancui/pca_colored_by_extreme_precip.pdf](figures/review_huancui/pca_colored_by_extreme_precip.pdf) — PCA scatter coloured by each state's $Q_{99}$ of spatial-max precipitation. State 7's cluster (bottom-right) glows brightest.

---

## Comment 2 — Why is the State 5 → State 3 transition not drawn in Fig. 7a?

> *About how to interpret the Markov transition matrix shown in Figure 7b, the value from state 4 to state 2 [→ State 5 → State 3 in the new labels] is 0.12 [actual matrix value 0.16], which is pretty high compared with other state transitions. Does it mean that it is very likely for an initial state of 4 to transit to state 2? If so, why is this transition link not indicated in Figure 7a? Is it because such transition is not particularly enhanced or suppressed in the dominant dynamic mode that you obtained from Section 3.2?*

**Response.** The reviewer's hypothesis is essentially correct. Two factors combine:

1. **State 5 (old State 4) is one of the rarely-occupied states** (π₅ ≈ 4.0%). The figure caption of Fig. 7 already states that transitions involving States 1 and 5 are omitted from panel (a) for visual clarity. They are kept in panel (b) — which is why the reviewer can still see the 0.16 entry — but they are suppressed in the latent-space arrow diagram, which is intended to highlight the *dominant* dynamical pathways.

2. **The reviewer's intuition is right about the eigenmode.** Looking at the leading non-trivial eigenvector quoted in §3.2,
   $$\mathbf{v}_2 = [-17\%, \; -29\%, \; +47\%, \; -76\%, \; -3\%, \; -68\%, \; +100\%],$$
   State 5 carries an amplitude of only $-3\%$ in $\mathbf{v}_2$ — effectively zero. State 5 is therefore *not* a participant in the dominant slow dynamical mode that organizes the chain into the two pathways discussed in §3.2. It is a low-occupancy "bystander" in the latent space — and so omitting it from the schematic in Fig. 7a does not hide any feature of the dominant dynamics.

The same conclusion is sharper if we look at the **unconditional flux** $\pi_5 \, T_{5j}$ rather than the conditional probability $T_{5j}$ (Fig. [intensity_organization_and_state5_flux.pdf](figures/review_huancui/intensity_organization_and_state5_flux.pdf), panel b):

| Transition $j$ | $T_{5j}$ (conditional) | $\pi_5 \, T_{5j}$ (unconditional flux) |
|---|---|---|
| 5 → 3 | **0.163** | 0.0065 |
| 5 → 6 | 0.059 | 0.0024 |
| 5 → 4 | 0.057 | 0.0023 |

Even the largest outgoing transition, $5 \to 3$ with $T = 0.163$, contributes only $\sim 0.65\%$ of all transitions globally. For comparison, the unconditional flux $\pi_4 \, T_{46}$ (State 4 → State 6, a within–weak-regime transition) is $\sim 1.5\%$, more than twice as large, despite a *smaller* conditional probability ($T_{46} = 0.076$). The arrow diagram is drawn from the *flux*, not from the matrix row — the reviewer's reading of the matrix is correct, but the matrix value alone overstates the dynamical importance of this transition.

**Action.** We will (a) keep the figure as is and (b) extend the caption of Fig. 7 to make the criterion explicit: "Arrows in panel (a) are proportional to the unconditional flux $\pi_i T_{ij}$ rather than $T_{ij}$ alone; transitions involving States 1 and 5 are omitted because $\pi_1, \pi_5 < 5\%$ and both carry $|\mathbf{v}_2| < 0.2$, contributing negligibly to the dominant dynamical mode." `[→]`

**Supporting figure produced**: [figures/review_huancui/intensity_organization_and_state5_flux.pdf](figures/review_huancui/intensity_organization_and_state5_flux.pdf), panel **(c)**.

---

## Comment 3 — What is the ultimate purpose of this framework?

> *A final question is that, from reading the manuscript, it is not very clear to me what the ultimate purpose of this framework although it is very cool. Is it mainly to help understand MCS processes (as you currently showed) or can it be used to build a simplified model to predict convective systems? Maybe to expand a little bit on how to use this framework?*

**Response.** A fair question — the manuscript currently emphasises the *diagnostic* side and underplays the prospective uses. We will add a "Use cases of the framework" subsection at the end of §4 (Discussion) along the following lines:

The framework has **three distinct uses**, which we plan to make explicit in the revised text:

1. **Process understanding (current paper).** The framework provides a *dynamically meaningful* coordinate system in which to characterise convective variability. Trajectories along PCA1 and PCA2 separate convective lifecycles by intensity and organization; the Markov chain identifies which sequences of states are dynamically preferred; and the residual non-Markovianity diagnoses the presence of internal feedbacks (cold pools in our case, but the methodology generalises). This is the use demonstrated in the current paper. It is non-trivial because it lets us *test* hypotheses about convective memory directly from CRM output, in a low-dimensional space where statistical questions have well-defined answers — for example, "does cold-pool intensity carry information beyond the discrete state?" becomes the survival-stratification test of Fig.~8b.

2. **A learned stochastic parameterization.** The trained transition matrix $\mathbf{T}$ over the seven states is itself a candidate *simplified model* for convective evolution: given the current state, $\mathbf{T}$ predicts a probability distribution over the next state. Because the latent representation is decoded by the VAE back to a mass-flux profile, sampling the chain yields a *generative* model — synthetic trajectories of convective mass flux that respect the statistics of the CRM. This is the data-driven alternative to prescribed multi-cloud Markov schemes (\cite{khouider2010stochastic}) that the abstract already alludes to but does not develop. Adding cold-pool intensity to the state description (per §3.3) is then a constructive recipe for *improving* this stochastic emulator — a path the manuscript points to but stops short of executing.

3. **Diagnosing memory in existing parameterizations.** The autocorrelation / survival diagnostics we develop (Fig. 8) are *transferrable*: they apply equally to any convection scheme that can be queried under similar idealized conditions. Following the philosophy of \cite{hwong2023assessing}, our seven-state Markov reference provides a quantitative null against which a parameterization's memory structure can be compared. A scheme that lacks cold-pool memory will fail to reproduce the survival-curve heterogeneity of Fig.~8b; a scheme with too much memory will overshoot it.

We will reframe the closing paragraphs of §4 to flag the framework's use *prospectively* as a tool for building and testing stochastic convective representations, rather than only as a diagnostic of the current simulations.

**Action.** `[→]` Add a short subsection (1 paragraph, ~150 words) at the end of §4 titled *Three uses of the framework*: process diagnosis, learned parameterization, and memory-testing of existing schemes. Plus 1–2 sentences in the abstract and introduction signalling these.

---

## Summary of actions to take

| Item | Type | Where | Status |
|---|---|---|---|
| Reframe PCA1 as a *convective-life-cycle / extreme-precipitation* axis; contrast per-state mean vs extreme contributions | text + 1 panel | §3.1 | `[→]` |
| Note that Fig. 7a arrows are proportional to flux $\pi_i T_{ij}$, not $T_{ij}$ | caption | Fig. 7 caption | `[→]` |
| Cite reasons for omitting States 1 and 5 (low π, small $\mathbf{v}_2$ amplitude) | caption | Fig. 7 caption | `[→]` |
| Add subsection *Three uses of the framework* | text | end of §4 | `[→]` |
| Reflect the three uses in abstract & introduction | text | abstract + §1 | `[→]` |
| Produce supporting figure (intensity-vs-organization + state-5 flux) | figure | [figures/review_huancui/intensity_organization_and_state5_flux.pdf](figures/review_huancui/intensity_organization_and_state5_flux.pdf) | `[✓]` |
| Script that produces the figure | code | [review_huancui_figure.py](review_huancui_figure.py) | `[✓]` |
