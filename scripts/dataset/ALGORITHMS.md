# CellARc Episode Generation Algorithms

This note explains the algorithms and supporting libraries that power the dataset scripts. The code lives under `cellarc/generation`, while the CLI tooling in `scripts/dataset/*.py` simply wires these components together. When you need to reason about how episodes are produced--or to extend the generator--start here.

---

## Core execution backend (`cellarc/generation/cax_runner.py`)

- **Packages**: [`jax`](https://github.com/google/jax), [`flax`](https://github.com/google/flax), and the lightweight [`cax`](https://github.com/deepmind/cax) framework provide the differentiable cellular-automata runtime. We wrap them inside `AutomatonRunner`, which materialises a deterministic rule-table automaton as a `cax.core.ComplexSystem`.
- **Dense rule tables**: Irrespective of how a rule is specified (sparse dict, procedural generator, etc.), `rule_table.ensure_dense_rule_table` converts it into a contiguous array of size `k^(2r+1)`. `_transition_arrays` also precomputes the base powers needed to turn an `(2r+1)`-wide neighbourhood into an index in that array.
- **Device placement**: By default we pin execution to CPU (`CELLARC_FORCE_CPU=1`) because the short CA traces used during sampling do not benefit from GPU launch overhead. Set `CELLARC_FORCE_CPU=0` to allow JAX to pick the fastest device.
- **Evolving a rule**: `AutomatonRunner.evolve(init_state, timesteps, return_history)` emits either the final state or the full space-time diagram, which downstream samplers slice into train/query/solution segments.

---

## Rule generation (`cellarc/generation/rules.py`)

`make_pool.py` and friends never construct rules directly; instead they ask the family-specific helpers listed below to synthesise a table and report the achieved Langton lambda:

| Family helper | Behaviour summary |
|---------------|-------------------|
| `rule_table_random_lambda` | Draws each neighbourhood outcome independently while matching a target lambda (fraction of non-quiescent outputs). |
| `rule_table_totalistic` | Depends only on the sum of neighbourhood symbols. |
| `rule_table_outer_totalistic` / `rule_table_outer_inner_totalistic` | Split the centre from the outer shell, letting lambda be targeted per outer sum. |
| `rule_table_threshold` | Chooses per-state thresholds and pair modes (majority-like vs random pairs) over the outer shell. |
| `rule_table_linear_mod_k` | Builds an affine function modulo `k` with sparse coefficients and optional bias. |
| `rule_table_cyclic_excitable` / `rule_table_permuted_totalistic` | Provide specialised excitable / permuted behaviours for diversity. |

Every helper returns a `DenseRuleTable`, the realised lambda, the quiescent state, and any extra family parameters that we store inside the episode metadata for auditing.

---

## Episode sampling (`cellarc/generation/sampling.py`)

`generate_dataset_jsonl` repeatedly calls `sample_task`, which performs the following steps for each episode:

1. **Alphabet / receptive-field selection**  
   - Sample an alphabet size `k` from the requested range (biasing toward smaller `k` for efficiency).  
   - Estimate a target window size `W` from the total supervision budget and clamp it to `2 * max_radius * max_steps + 1`.  
   - Solve `2 * r * t + 1 = W` via `choose_r_t_for_W`, returning an `(r, t)` pair consistent with the caps.

2. **Rule instantiation**  
   - Sample a family using the configured mixture (see `make_pool.py` defaults) and call the respective generator.  
   - Record Langton lambda, entropy bins, morphology descriptors, and fingerprints (`fingerprints.py`) so we can deduplicate later.

3. **Space-time rollout**  
   - Build a de Bruijn cycle `de_bruijn_cycle(k, W)` (details below) and evolve the CA for `t + 1` steps.  
   - Optional constructions:  
     - `cycle`: always uses the steady-state cycle row (`tau=0`).  
     - `unrolled`: releases multiple time slices from the de Bruijn trajectory.  
     - `hybrid`: mixes `tau=0` with positive offsets to widen the context.

4. **Coverage-aware train/query extraction**  
   - Compute training spans by sliding windows of length >= `W` over the de Bruijn cycle. `coverage_fraction` and `coverage_mode` decide how densely these spans tile the cycle (`chunked` = random starts, `uniform` = evenly spaced + jitter).  
   - For unrolled/hybrid episodes we also sample time offsets (`tau`) so the inputs come from different portions of the history.  
   - Queries either come from a fresh random state (`cycle`) or from another slice of the rolled-out history (with matching output taken `t` steps later).  
   - Coverage metadata (`meta.coverage`) reports how many unique windows appear in the training context and whether the query is guaranteed to lie inside the revealed region.

5. **Metadata & fingerprints**  
   - `induced_tstep_fingerprint` hashes the `(rule, k, r, t)` combination to enforce uniqueness when `unique_by="tstep"`.  
   - Optional rule-table payloads are serialised with `serialization.serialize_rule_table` (base64 + schema version).

All of the heavy lifting--entropy/AMI metrics, morphology, and randomness--is handled inside the sampler; the CLI scripts only tweak the knobs (family mix, coverage bounds, etc.).

---

## De Bruijn coverage scheme (`cellarc/utils.py`)

- `de_bruijn_cycle(k, n)` returns a length-`k^n` cyclic sequence in which every length-`n` window appears exactly once. We treat it as a wrap-around tape (`ring_slice`) so the sampler can grab arbitrary contiguous spans without worrying about boundaries.
- Using de Bruijn cycles ensures:
  1. **Uniform coverage**: every possible neighbourhood of width `W` is represented, making it easy to reason about coverage fractions (we simply count how many unique windows a set of spans touches).  
  2. **Leak-free queries**: because the query either comes from the same cycle (after reshuffling) or from an unrolled history, we can precisely control whether its windows appear in training.
- The coverage metadata labelled `scheme: "de_bruijn_subcover"` in every episode reflects this construction. When `query_within_coverage=True` we partition the cycle deterministically so the query is guaranteed to lie inside the revealed region; otherwise the query may probe unseen windows and we track that via `meta.query_window_coverage_*`.

---

## Ancillary components

- **Ring slicing (`cellarc/generation/helpers.py`)** - `ring_slice` wraps indices mod cycle length so window extraction works for both `cycle` and `unrolled` constructions.  
- **Metrics (`metrics.py`, `morphology.py`)** - After generating a rule we roll it forward on random states (width 30, horizon 256) to estimate average cell entropy, mutual information, Derrida-like coefficients, absorbing/periodic flags, etc. These features back the stratified down-sampler and the coverage-aware split heuristics.  
- **Fingerprints (`fingerprints.py`)** - Provide reproducible `meta.fingerprint` / `meta.probe_fingerprint` identifiers for deduplication and dataset IDs.  
- **Serialization (`serialization.py`)** - Embeds rule tables and metadata schema versions so that downstream tooling (HF datasets, reconstruction scripts) can round-trip episodes without rerunning the generators.

Together these pieces fully specify how every episode in `cellarc_100k(_meta)` was created, independent of the CLI wrappers. Consult this document when auditing a release or when you need to modify the generation logic beyond what the scripts expose as flags.
