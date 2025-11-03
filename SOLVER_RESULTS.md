# Solver Accuracy Correlates

## Strongest predictors

| Feature | Pearson r (accuracy) | Spearman ρ (accuracy) | How it is computed |
| --- | --- | --- | --- |
| `query_window_coverage_weighted` | 0.94 | 0.94 | For every length‑W window in the query sequence, count how often it appears (frequency). Sum those frequencies over the windows already seen in training and divide by the total number of query windows: \\( \\frac{\\sum_{w \\in Q \\cap T} \\text{freq}_Q(w)}{\\sum_{w \\in Q} \\text{freq}_Q(w)} \\). |
| `query_window_coverage_unique` | 0.93 | 0.93 | Ratio of distinct query windows that were observed during training: \\( \\frac{|Q \\cap T|}{|Q|} \\). |
| `query_window_avg_depth` | 0.62 | 0.86 | Average number of times the training set showed each window that later appears in the query: \\( \\frac{1}{|Q|} \\sum_{w \\in Q} \\text{freq}_T(w) \\). |
| `coverage_windows` | 0.82 | 0.82 | Total number of windows deliberately revealed by the sampler (independent of the query). |
| `coverage_windows / \\text{query_length}` | 0.32 | 0.32 | Normalised coverage count (proxy for general availability of context). |
| `lambda` | -0.18 | -0.20 | The CA’s Langton λ parameter from metadata; higher λ (chaotic rules) reduce accuracy. |
| `avg_cell_entropy` | -0.26 | -0.27 | Episode entropy from metadata; higher entropy correlates with more solver errors. |
| `ncd_train_query_solution` | -0.19 | -0.18 | Normalised compression distance (zlib) between flattened train+query sequence and the ground-truth solution. |

All metrics are computed per episode and the solver accuracy values are averaged over 10 stochastic runs (random backoff seeded by `episode_seed + run_idx`).

## Discussion

1. **Exact query coverage dominates** – The solver only fails when the query contains neighbourhoods it has never seen. Measuring coverage over the *actual query windows* (unique or weighted) directly captures that fact and lines up almost perfectly with accuracy. The weighted version is slightly stronger because repeated windows in the answer matter more.

2. **Depth helps when coverage is imperfect** – Even if a query window appears in training, seeing it many times (high average depth) stabilises the solver’s backoff by reinforcing the dominant output. Episodes with shallow coverage remain volatile.

3. **Global coverage is informative but insufficient** – `coverage_windows` and its normalised variants correlate well with success, yet their signal is weaker than query-specific overlap. Two episodes can expose the same number of windows globally but differ dramatically in how well those windows align with the held-out query.

4. **Dynamics shape the difficulty** – Higher λ, higher entropy, and larger rule alphabets push the solver into chaotic regimes where unseen windows are more common, lowering accuracy even when nominal coverage looks high. These features act as rough “risk” indicators.

5. **Distributional mismatch hurts** – The negative correlation from NCD shows that when the flattened solution looks unlike the glimpsed trajectories, the solver is forced to guess, again pointing to the importance of query-aligned observations.

Overall, the most actionable predictor is the proportion of query windows seen in training. That statistic can be derived before running the solver and used either to short-circuit hopeless cases or to trigger more aggressive data augmentation for low-coverage episodes. The other metadata features explain why certain episodes enter that low-coverage regime (rule complexity, entropy, λ), highlighting where the dataset pushes the solver beyond its ideal assumption of perfect de Bruijn coverage.
