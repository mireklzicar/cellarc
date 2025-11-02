# cellarc

## Generation of the dataset

### 1. Making a pool

Target: generate a pool of diverse CAs without many constraints. That gives headroom to filter the quiescent, unsolvable or uninteresting ones and still land cleanly on 100K/1K/1K.

```
# optional: faster on GPU if JAX/CUDA is installed
export CELLARC_FORCE_CPU=0

python scripts/make_pool.py \
  --outdir artifacts/pool \
  --per-shard 10000 \
  --shards 6 \
  --family-bonus 8 \
  --k-min 2 --k-max 10 \
  --max-radius 3 --max-steps 5 \
  --train-examples 5 --avg-train-len 48 \
  --balance-by lambda \
  --unique-by tstep \
  --constructions cycle unrolled hybrid \
  --coverage-modes chunked uniform \
  --sample-timeout 0.1 \
  --no-complexity \
  --no-morphology
```

That yields ~(6 shards × 3 constructions × 2 cov modes + 8 families × 3 constructions) × 10K ≈ 180K episodes, deduped across shards by t‑step fingerprint.

Performance tweaks:
- `--sample-timeout 0.1` skips any single CA episode that takes longer than the budget (requires Unix SIGALRM support).
- `--no-complexity` and `--no-morphology` remove the most expensive analytics passes when you just need raw rollouts quickly.
- Increase or decrease `--max-attempts-per-item` to adjust the retry budget per accepted sample (default 200).
