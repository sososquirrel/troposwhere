# Annotated bibliography

One-line annotations for every paper cited in [main.tex](manuscript/main.tex) and [supplementary.tex](manuscript/supplementary.tex). Grouped by the role they play in our framing.

Format: `[bibkey] Author Year` — one-line annotation.

---

## MCS observations & datasets (Introduction — convection complexity)

- **[hu2020observed]** Hu et al. 2020 — uses GPM-DPR observations to show that MCS and non-MCS warm-season rainfall have distinct microphysical and vertical-structure signatures.
- **[fiolleau2024database]** Fiolleau et al. 2024 — builds a multi-satellite database of tropical deep convective systems by intercalibrated tracking, supplying observational priors on MCS lifetime/structure.
- **[feng2025mesoscale]** Feng et al. 2025 — intercomparison of MCS tracking algorithms (MCSMIP) showing that tracker choice substantially modulates inferred MCS statistics.
- **[chakraborty2023overview]** Chakraborty 2023 — review of mesoscale convective systems: global climatology, satellite signatures, and predictability gaps.

## Convective response to warming & extremes (Introduction — why convection matters)

- **[yang2023impact]** Yang et al. 2023 — high-resolution downscaling shows that summertime MCS frequency and intensity increase under warming over the U.S., contributing to extreme rainfall change.
- **[bolot2025no]** Bolot et al. 2025 — argues that *individual* deep convective systems do not weaken under warming despite mean-state circulation slowdown, focusing the response onto extremes.
- **[prein2022towards]** Prein et al. 2022 — proposes a unified simulation setup to compare mid-latitude and tropical MCSs across model resolutions.
- **[MullerYukari19]** Muller & Yukari 2020 — review of precipitation extremes under warming, with emphasis on how *internal variability* of convective systems can dominate the climatological signal.
- **[singh2012upward]** Singh & O'Gorman 2012 — uses idealized RCE simulations to show that the convective vertical mass-flux profile shifts upward with warming, modulating extreme precipitation.
- **[abramian2023extreme]** Abramian et al. 2023 — characterizes extreme precipitation in tropical squall lines and shows that vertical-mass-flux structure controls the rain-rate distribution.
- **[wang2022convective]** Wang et al. 2022 — uses GoAmazon and MC3E radar to document updraft/downdraft properties of continental MCSs across regimes.
- **[li2025bimodal]** Li et al. 2025 — shows that tropical precipitation extremes exhibit a *bimodal* vertical-motion structure (bottom-heavy vs top-heavy) with distinct rain responses.

## Stochastic multicloud / Markov models for convection (Introduction — prior reduced-order work)

- **[khouider2006multicloud]** Khouider & Majda 2006 — original multicloud convective parameterization with three prescribed cloud types (congestus, deep, stratiform) and crude vertical structure.
- **[majda2008stochastic]** Majda et al. 2008 — adds stochastic momentum transport to the multicloud framework, making the convective response probabilistic rather than deterministic.
- **[khouider2010stochastic]** Khouider et al. 2010 — fully stochastic multicloud model: convective columns evolve as a Markov chain on a lattice with environment-dependent transition probabilities. **Direct precursor to our framework, but with prescribed (not learned) states.**

## Koopman operator theory (Introduction — theoretical scaffolding)

- **[koopman1931hamiltonian]** Koopman 1931 — original paper introducing the linear operator that propagates observables of a nonlinear dynamical system; foundation of Koopman analysis.
- **[brunton2021modern]** Brunton et al. 2021 — modern review of Koopman operator theory and its data-driven approximations (DMD, EDMD, deep variants).
- **[lusch2018deep]** Lusch et al. 2018 — learns Koopman observables with deep autoencoders so dynamics are *linear by construction* in the learned latent space. **Methodological template for our encoder-decoder design.**
- **[nathaniel2026generative]** Nathaniel et al. 2026 — extends Koopman-style emulation to chaotic / stochastic systems with coherent priors. **Closest contemporary work; same lineage as ours, but for chaotic emulation rather than discrete-state Markov reduction.**

## Isentropic analysis (Methods — input representation)

- **[pauluis2013isentropic]** Pauluis & Mrowiec 2013 — introduces isentropic analysis of convective motions: reorganizes vertical mass flux along moist static energy / equivalent potential temperature axes. **Source of our input representation.**
- **[mrowiec2016isentropic]** Mrowiec et al. 2016 — applies isentropic analysis to a simulated hurricane, demonstrating that coherent structures emerge clearly in $(h,z)$ space.
- **[chen2022crucial]** Chen et al. 2022 — argues that mesoscale convective systems are responsible for a disproportionate fraction of vertical mass and water transport, validating MCS-scale resolution as the analysis target.
- **[chen2018atmospheric]** Chen et al. 2018 — uses isentropic analysis on an MJO event to separate convective from gravity-wave overturning by their thermodynamic signature.
- **[chen2020significant]** Chen et al. 2020 — quantifies mesoscale overturning contribution to tropical mass and energy budgets using isentropic analysis.

## Cluster / partition-based reduced models (Methods — discrete state assignment)

- **[kaiser2014cluster]** Kaiser et al. 2014 — cluster-based reduced-order modelling of a mixing layer: partition the state space and model the dynamics as a Markov chain over clusters. **Direct conceptual precedent for our K=7 partition + transition matrix.**
- **[souza2024representing]** Souza 2024 — partition-based representation of turbulent statistics; provides theoretical grounding for using partitions of state space to capture stochastic dynamics.

## Squall lines / shear / convective organization (Methods — simulation setup)

- **[muller2013impact]** Muller 2013 — shows that convective organization sharply increases precipitation extremes in cloud-resolving simulations. **Source of the shear configuration we use.**
- **[rotunno1988theory]** Rotunno, Klemp & Weisman 1988 — classical RKW theory: optimal squall-line longevity occurs when cold-pool circulation balances low-level shear.
- **[houze1977structure]** Houze 1977 — observational characterization of squall-line structure and dynamics (the canonical descriptive paper).
- **[robe2001effect]** Robe & Emanuel 2001 — quantifies how vertical wind shear modulates radiative-convective equilibrium and convective organization.
- **[abramian2022shear]** Abramian et al. 2022 — links shear magnitude to squall-line orientation and intensity, motivating the shear value we choose for our simulations.

## Atmospheric model (Methods — SAM)

- **[khairoutdinov2003]** Khairoutdinov & Randall 2003 — describes the System for Atmospheric Modeling (SAM) cloud-resolving model used here; documents formulation, microphysics, and validation against the ARM 1997 IOP.

## Convective memory / quasi-equilibrium parameterizations (§3.3 — Markovianity discussion)

- **[zhang2003convective]** Zhang 2003 — observational test of quasi-equilibrium in the tropical western Pacific: CAPE adjusts rapidly to large-scale forcing, supporting a *memoryless* closure in tropical maritime regimes.
- **[bechtold2014representing]** Bechtold et al. 2014 — modifies the ECMWF IFS convection scheme to handle equilibrium (tropical) vs non-equilibrium (mid-latitude) convection with different closure timescales — i.e. *one timescale is not enough*.
- **[piriou2007approach]** Piriou et al. 2007 — first introduction of explicit prognostic memory variables (cloud water, precipitation) inside a convection scheme.
- **[gerard2009cloud]** Gerard et al. 2009 — meso-γ operational scheme (ALARO) extending Piriou with prognostic convective updraft mass flux; relevant for resolutions near our ~1 km grid (the "grey zone").
- **[mapes2011parameterizing]** Mapes & Neale 2011 — adds an "organization" prognostic variable to a CAM convection scheme to lag the convective response. **Closest spiritual precursor to our work — they prescribe an organization axis; we learn one (PCA 2).**
- **[hwong2023assessing]** Hwong et al. 2023 — idealized switch-off forcing test that *quantifies* memory of convection schemes by comparing them to CRM truth. **Validates our use of autocorrelation as a memory diagnostic.**
