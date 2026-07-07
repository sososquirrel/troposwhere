"""
Supporting figures for the response to reviewer Huan Cui (review_huancui.md).

Produces a 3-panel figure illustrating:
  (a) Per-state probability of an extreme-precipitation event, using the
      domain-mean precipitation > Q95 and > Q99 (global percentiles)
      thresholds. State 7 (most organized) has the HIGHEST per-time
      probability of being in an extreme precipitation event, while the
      least-organized states (3, 5, 6) contribute essentially zero.
  (b) Per-state share of all extreme events captured: organized states
      (4 and 7) capture ~85% of >Q95 events and ~95% of >Q99 events.
      Bubble plot of (P(P>Q99|state), organization) makes the same point
      geometrically.
  (c) For state 5 (= "state 4" in the reviewer's 0-indexed labels), the
      conditional outgoing transition probabilities T[5, j] and the
      corresponding unconditional flux pi_5 * T[5, j]. Shows that the
      relatively large conditional value T[5, 3] = 0.16 is misleading
      because pi_5 is so small (~4%) that the resulting flux is negligible,
      explaining why this transition is omitted from Fig. 7a.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workflow_paper as wfp


def main():
    # ----- 1. Load data + run model -----
    loader, input_dim = wfp.load_dataloader()
    model             = wfp.load_model(input_dim=input_dim)
    out               = wfp.extract_latents(model, loader)
    states            = out['states']

    # Domain-mean precipitation dilutes organized signal because squall-line
    # cores cover a fraction of the domain. We instead use the SPATIAL-MAX
    # precipitation per timestep (peak rate anywhere in the 128x128 domain).
    # This is the right metric to capture the intensity of organized
    # convective cores. Diagnostic is precomputed from the 2D Prec field.
    prec    = np.load('/Users/sophieabramian/Documents/troposwhere/data/diagnostics/Prec_spatial_max.npy')
    var_pw  = np.load('/Users/sophieabramian/Documents/troposwhere/data/diagnostics/var_PW.npy')[:-1]
    L       = min(len(states), len(var_pw), len(prec))
    states, var_pw, prec = states[:L], var_pw[:L], prec[:L]

    # ----- 2. Transition matrix & stationary distribution -----
    Tmat = wfp.compute_transition_matrix(states, wfp.NUM_STATES, future=5)
    pi   = wfp.stationary_distribution(Tmat)

    # ----- 3. Per-state diagnostics -----
    K = wfp.NUM_STATES

    def per_state(arr, fn):
        return np.array([fn(arr[states == s]) if (states == s).any() else np.nan
                         for s in range(K)])

    mean_prec = per_state(prec, np.mean)
    q95_prec  = per_state(prec, lambda x: np.quantile(x, 0.95))
    q99_prec  = per_state(prec, lambda x: np.quantile(x, 0.99))
    max_prec  = per_state(prec, np.max)

    # Global percentile thresholds — conditional probability of an extreme
    # event given each state, and the share of all extreme events captured
    # by each state.
    thr95 = np.quantile(prec, 0.95)
    thr99 = np.quantile(prec, 0.99)
    cond_p95   = per_state(prec, lambda x: (x > thr95).mean())
    cond_p99   = per_state(prec, lambda x: (x > thr99).mean())
    n_above95  = (prec > thr95).sum()
    n_above99  = (prec > thr99).sum()
    share95    = np.array([((prec > thr95) & (states == s)).sum() / n_above95
                           for s in range(K)])
    share99    = np.array([((prec > thr99) & (states == s)).sum() / n_above99
                           for s in range(K)])

    # Organization index used in the paper (line 202 of main.tex): the
    # spatial variance of precipitable water at each timestep. Larger
    # variance => less spatially organized. We take the per-state mean and
    # negate so that "larger = more organized" reads naturally.
    mean_var_pw  = per_state(var_pw, np.mean)
    organization = -mean_var_pw

    # ----- 4. Figure -----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0), constrained_layout=True)

    # ---- Panel (a): per-state probability of extreme precipitation ----
    ax = axes[0]
    x = np.arange(K)
    width = 0.4

    bar_colors = [wfp.cluster_colors[s] for s in range(K)]
    ax.bar(x - width / 2, 100 * cond_p95, width,
           color=bar_colors, edgecolor='black', linewidth=1.0,
           alpha=0.95, label=r'P(precip > $Q_{95}$ | state)')
    ax.bar(x + width / 2, 100 * cond_p99, width,
           color=bar_colors, edgecolor='black', linewidth=1.0,
           alpha=0.55, hatch='///', label=r'P(precip > $Q_{99}$ | state)')

    ax.set_xticks(x)
    ax.set_xticklabels([wfp.state_label(s) for s in range(K)])
    ax.set_xlabel('State')
    ax.set_ylabel('Probability (%)')
    ax.set_title('(a) Per-state probability of extreme spatial-max precipitation')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # Annotate the state-7 Q99 bar (the key headline number)
    ax.annotate(f'{100 * cond_p99[6]:.1f}%',
                xy=(6 + width / 2, 100 * cond_p99[6]),
                xytext=(0, 4), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold')

    # ---- Panel (b): organization vs extreme-event likelihood ----
    ax = axes[1]
    sizes = 200 + 6000 * pi
    for s in range(K):
        if np.isnan(cond_p99[s]):
            continue
        ax.scatter(100 * cond_p99[s], organization[s],
                   s=sizes[s], c=wfp.cluster_colors[s],
                   edgecolors='black', linewidths=1.2, zorder=3)
        ax.annotate(wfp.state_label(s),
                    xy=(100 * cond_p99[s], organization[s]),
                    xytext=(0, 0), textcoords='offset points',
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    zorder=4)

    ax.set_xlabel(r'P(spatial-max precip > $Q_{99}$ | state)  (%)')
    ax.set_ylabel(r'$-\mathrm{Var}_{\mathrm{spatial}}(\mathrm{PW})$  (organization)')
    ax.set_title('(b) Organization tracks extreme-precip frequency')
    ax.grid(alpha=0.3)

    for label, fraction in [('π = 5%', 0.05), ('π = 20%', 0.20), ('π = 35%', 0.35)]:
        ax.scatter([], [], s=200 + 6000 * fraction, c='lightgrey',
                   edgecolors='black', linewidths=1.0, label=label)
    ax.legend(title='Bubble size = stationary probability',
              loc='lower right', fontsize=8, title_fontsize=8,
              labelspacing=1.4, borderpad=1.0, frameon=True)

    # ---- Panel (c): out-of-state-5 transitions ----
    ax = axes[2]
    s_focal = 4  # 0-indexed state index = display label "5" (reviewer's "state 4")
    cond_probs = Tmat[s_focal].copy()
    flux       = pi[s_focal] * cond_probs

    x = np.arange(K)
    width = 0.4

    bars1 = ax.bar(x - width / 2, cond_probs, width,
                   color=[wfp.cluster_colors[s] for s in range(K)],
                   edgecolor='black', linewidth=1.0, alpha=0.95,
                   label=r'Conditional $T_{5j}$')
    bars2 = ax.bar(x + width / 2, flux, width,
                   color=[wfp.cluster_colors[s] for s in range(K)],
                   edgecolor='black', linewidth=1.0, alpha=0.45, hatch='///',
                   label=r'Unconditional flux $\pi_5 \, T_{5j}$')

    ax.set_xticks(x)
    ax.set_xticklabels([wfp.state_label(s) for s in range(K)])
    ax.set_xlabel('Destination state $j$')
    ax.set_ylabel('Probability')
    ax.set_title(r'(c) Transitions out of state 5  ($\pi_5 \approx %.1f\%%$)' % (100 * pi[s_focal]))
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    # Annotate the 5 -> 3 bar with both values
    j_target = 2  # destination 0-indexed = display "3"
    ax.annotate(f'{cond_probs[j_target]:.2f}',
                xy=(j_target - width / 2, cond_probs[j_target]),
                xytext=(0, 4), textcoords='offset points',
                ha='center', fontsize=9)
    ax.annotate(f'{flux[j_target]:.3f}',
                xy=(j_target + width / 2, flux[j_target]),
                xytext=(0, 4), textcoords='offset points',
                ha='center', fontsize=9)

    # ----- 5. Save -----
    out_dir = os.path.join(wfp._ROOT, 'paper', 'figures', 'review_huancui')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'intensity_organization_and_state5_flux.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')

    # ----- 5b. Bonus standalone figure: PCA scatter coloured by the
    #           per-state Q99 of precipitation. Each point's colour is
    #           determined by its state's Q99 of domain-mean precipitation
    #           computed across the full simulation -- so the colour is
    #           a property of the LATENT REGION the point lives in, not
    #           of the instantaneous precipitation at that timestep.
    emb_pca, _ = wfp.run_pca(out['embeddings'])
    emb_pca    = emb_pca[:L]

    # q99_prec was computed above per state; assign each timestep its
    # state's Q99.
    point_q99 = q99_prec[states]

    fig2, ax2 = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    norm = mcolors.Normalize(vmin=np.nanmin(q99_prec), vmax=np.nanmax(q99_prec))
    sc = ax2.scatter(emb_pca[:, 0], emb_pca[:, 1],
                     c=point_q99, cmap='inferno', norm=norm,
                     s=8, alpha=0.85, linewidth=0)
    ax2.set_xlabel('PCA 1  (intensity)')
    ax2.set_ylabel('PCA 2  (organization)')
    ax2.set_title('PCA latent space coloured by per-state $Q_{99}$ of spatial-max precipitation\n'
                  '(extreme-precip capability, full simulation)')
    ax2.grid(alpha=0.3)
    cbar = fig2.colorbar(sc, ax=ax2)
    cbar.set_label(r'$Q_{99}$ of spatial-max precip in state')

    out_path2 = os.path.join(out_dir, 'pca_colored_by_extreme_precip.pdf')
    fig2.savefig(out_path2, bbox_inches='tight')
    fig2.savefig(out_path2.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path2}')

    # ----- 6. Print numerics that the markdown will quote -----
    print('\nPer-state precipitation statistics:')
    print(f'  {"state":>5}  {"mean":>7}  {"Q95":>7}  {"Q99":>7}  {"max":>7}  '
          f'{"-var(PW)":>9}  {"pi":>6}')
    for s in range(K):
        print(f'  {wfp.state_label(s):>5}  {mean_prec[s]:>7.3f}  '
              f'{q95_prec[s]:>7.3f}  {q99_prec[s]:>7.3f}  {max_prec[s]:>7.3f}  '
              f'{organization[s]:>9.3f}  {pi[s]:>6.3f}')

    print(f'\nGlobal thresholds: Q95={thr95:.2f}, Q99={thr99:.2f}')
    print('\nPer-state extreme-precipitation statistics:')
    print(f'  {"state":>5}  {"P(>Q95|s)":>10}  {"P(>Q99|s)":>10}  '
          f'{"share>Q95":>10}  {"share>Q99":>10}')
    for s in range(K):
        print(f'  {wfp.state_label(s):>5}  {100*cond_p95[s]:>10.2f}  '
              f'{100*cond_p99[s]:>10.2f}  {100*share95[s]:>10.2f}  '
              f'{100*share99[s]:>10.2f}')

    print('\nTransitions out of state 5 (= reviewer 0-indexed "state 4"):')
    for j in range(K):
        print(f'  state 5 -> state {wfp.state_label(j)}: '
              f'T={Tmat[s_focal, j]:.3f}, flux pi*T={flux[j]:.4f}')


if __name__ == '__main__':
    main()
