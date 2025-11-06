Here’s a tight, paper-ready figure plan for CellARC. Each item says what story it tells, what to plot, and where the numbers come from in this repo so it’s reproducible.

Core figures (main paper)
	1.	Pipeline diagram (end-to-end)
	•	Story: Transparent, reproducible data creation.
	•	What: A flowchart covering make_pool → pool_stats → pool_filtering → sanity_check → enrich_downsampled → split_pool → build_hf_dataset, with inputs/outputs and counts.
	•	Source: DATASET_CREATION.md, scripts/processing_pipeline.py.
	2.	Rule-space coverage
	•	Story: The benchmark spans alphabets, radii, and time-steps.
	•	What: Three histograms (or a small multiples row): alphabet_size k, radius r, steps t. Add a scatter of (r, t) with diagonal bands of constant window W=2rt+1 annotated.
	•	Source: artifacts/datasets/*/dataset_stats.json produced by build_hf_dataset.py.
	3.	λ × coverage grid (before/after downsampling)
	•	Story: Stratified selection achieves broad support, not just dense clusters.
	•	What: Two 2D heatmaps: bin counts over (Langton λ bin, observed coverage fraction bin) for raw pool vs. downsampled pool.
	•	Source: Counters emitted in scripts/pool_filtering.py (see assign_bins & stats printed), plus pool_stats.py if you log pre-filter.
	4.	Split shift overview
	•	Story: test_extrapolation is deliberately low-coverage/high-λ vs train/val/test_interp.
	•	What: Overlaid KDEs or faceted histograms per split for: query_window_coverage_weighted, lambda, avg_cell_entropy. A tiny table with split sizes.
	•	Source: After running scripts/resplit_dataset.py, these features are in each JSONL; sizes in build_hf_dataset.py stats.
	5.	Episode “postcards”
	•	Story: Qualitative feel for tasks.
	•	What: A 4×5 montage of episode cards spanning families and coverage regimes.
	•	How: scripts/plot_episode_cards.py --count 20 on each split (or a stratified pick).
	6.	Accuracy vs. query coverage (unique & weighted)
	•	Story: Coverage over actual query windows almost perfectly predicts success.
	•	What: Two panels: scatter with binned means & 95% CI for (accuracy, query_window_coverage_unique) and (accuracy, query_window_coverage_weighted). Add Pearson/Spearman insets.
	•	Source: Compute features with scripts/resplit_dataset.py, accuracies with scripts/evaluate_solver.py. Correlates summarized in SOLVER_RESULTS.md.
	7.	Calibration curve for exact match
	•	Story: A practical threshold—above X% weighted coverage, exact matches become likely.
	•	What: Probability of exact match vs. binned query_window_coverage_weighted; include reliability line.
	•	Source: Same metrics as (6).
	8.	Failure modes (qualitative)
	•	Story: Where and why the baseline fails.
	•	What: 4 examples with (train, query, solution, prediction) and small side-bars highlighting unseen query windows.
	•	Source: Collect failing IDs from evaluate_solver.py --show-failures.
	9.	Dynamics & morphology landscape
	•	Story: Dataset covers ordered ↔ chaotic regimes.
	•	What: Scatter of avg_cell_entropy vs. derrida_like, points colored by λ-bin or family; mark absorbing/periodic with shape.
	•	Source: Added by enrich_downsampled.py (morphology block) or in the extended splits.
	10.	Distribution shift pay-off

	•	Story: Accuracy drop from interpolation to extrapolation.
	•	What: Bars (±CI) for exact-match and Hamming accuracy on val, test_interpolation, test_extrapolation. Optionally a paired dot plot per split.
	•	Source: evaluate_solver.py.

Secondary figures (nice-to-have or appendix)
	11.	Family mix per split

	•	What: Stacked bars of meta.family per split.
	•	Source: dataset_stats.json and/or split JSONL.

	12.	Alphabet size vs. window size

	•	What: Heatmap of counts over (k, W), with marginal histograms.
	•	Source: Split JSONL meta (alphabet_size, window).

	13.	Coverage windows relative to query length

	•	What: Histogram of coverage_windows / query_length per split—read as “context sufficiency”.
	•	Source: Split JSONL meta (coverage.windows, query length).

	14.	NCD vs. accuracy

	•	What: Scatter/binned curve of ncd_train_query_solution vs. accuracy (negative correlation).
	•	Source: resplit_dataset.py adds NCD; evaluate_solver.py adds accuracy.

	15.	Correlation heatmap

	•	What: Pearson and Spearman matrices of solver accuracy vs. key features (query_window_*, coverage_windows, lambda, avg_cell_entropy, ncd_*, alphabet_size).
	•	Source: Combine outputs from (6), (9), (14); numbers echo SOLVER_RESULTS.md.

	16.	Reconstruction integrity

	•	What: Tiny bar or table: % rule_fingerprint matches and % induced_tstep_fingerprint matches during enrichment (should be 100%); any mismatches counted as 0.
	•	Source: Exceptions in generation/reconstruction.py; log counts while running enrich_downsampled.py.

	17.	Generation efficiency

	•	What: Attempts vs. accepted items, timeouts, and average attempts/item per construction or family.
	•	Source: Stats returned by generate_dataset_jsonl (timeouts, attempt budgets); print or log during pool build and aggregate.

	18.	Signature embedding

	•	What: 2-D UMAP/t-SNE of vectors from cellarc/signatures.compute_signature, colored by family or split.
	•	Source: Run signatures.py over any split; store 2D coords.

Minimal plotting specs (so figures are consistent)
	•	Use the split JSONL from cellarc_100k_meta for meta-rich plots; when speed matters, Parquet mirrors are identical.
	•	For accuracy figures, average over multiple solver runs or report seed; evaluate_solver.py can be adapted to repeat with different rng_seed offsets (baseline’s backoff randomness).
	•	Always label window size as W=2r\,t+1; when binned, report bin edges in captions.
	•	Color palette: families (categorical) and λ-bins (ordered) need distinct schemes; keep the same mapping across figures.
	•	Confidence bands: bootstrap or binomial proportion intervals for exact-match; standard error for mean Hamming.

This set gives you: provenance (pipeline), coverage of the rule/dynamics space, split shift, qualitative texture, and a causal-looking story linking query coverage to solver success—plus appendices that make reviewer’s eyebrows settle. When you’re ready, I can sketch a small plotting harness that reads the JSONL/Parquet and spits out each figure with consistent styling.