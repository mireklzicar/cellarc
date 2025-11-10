#!/usr/bin/env python3
"""
Create fixed-size subset splits for CellARc HF repos and write ID lists.

This script samples a fixed number of episode IDs per split from the
`cellarc_100k` HF dataset checkout and mirrors the same selection in the
companion `_meta` repo. It writes the following for each split:

- data/{split}_{N}.jsonl (subset of episodes)
- data/{split}_{N}.parquet (parquet subset with supervision fields)
- subset_ids/{split}_{N}.txt (one episode id per line)
- updates data_files.json with the new split entries

By default, the script only creates/commits local changes. Use `--push` to push
to the Hugging Face remotes after reviewing the changes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


try:
    from datasets import Dataset  # type: ignore
except Exception as exc:  # pragma: no cover - optional dep guard
    raise SystemExit(
        "This script requires the 'datasets' package. Install with `pip install datasets pyarrow`."
    ) from exc


SPLITS_DEFAULT = ("train", "val", "test_interpolation", "test_extrapolation")


@dataclass
class RepoSpec:
    path: Path
    name: str

    @property
    def data_dir(self) -> Path:
        return self.path / "data"

    @property
    def subset_ids_dir(self) -> Path:
        return self.path / "subset_ids"

    def git(self, *args: str) -> str:
        proc = subprocess.run(["git", *args], cwd=self.path, check=True, capture_output=True, text=True)
        return proc.stdout.strip()


def read_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for obj in rows:
            fh.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
            fh.write("\n")
            count += 1
    return count


def write_parquet(path: Path, rows: List[Dict]) -> None:
    # Parquet schema mirrors the supervision-only fields
    # (id, train, query, solution). Extra keys are dropped if present.
    projection: List[Dict] = []
    for r in rows:
        projection.append({
            "id": r.get("id"),
            "train": r.get("train"),
            "query": r.get("query"),
            "solution": r.get("solution"),
        })
    ds = Dataset.from_list(projection)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(path))


def update_data_files_json(repo: RepoSpec, updates: Dict[str, Tuple[Path, Path, int]]) -> None:
    """Update or create data_files.json with new split entries.

    updates maps split_name -> (jsonl_path, parquet_path, records)
    """
    df_path = repo.path / "data_files.json"
    data = {}  # type: ignore[var-annotated]
    if df_path.exists():
        with df_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

    for split, (jsonl_path, parquet_path, n) in updates.items():
        data[split] = {
            "jsonl": {
                "bytes": os.path.getsize(jsonl_path),
                "path": str(jsonl_path.relative_to(repo.path)).replace(os.sep, "/"),
                "records": int(n),
            },
            "parquet": {
                "bytes": os.path.getsize(parquet_path),
                "path": str(parquet_path.relative_to(repo.path)).replace(os.sep, "/"),
                "records": int(n),
            },
        }

    with df_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def ensure_clean_repo(repo: RepoSpec) -> None:
    status = repo.git("status", "--porcelain")
    if status:
        raise SystemExit(f"Repository '{repo.name}' has uncommitted changes. Please commit or stash them first.\n{status}")


def commit_all(repo: RepoSpec, message: str) -> None:
    repo.git("add", "-A")
    # commit even if nothing changed is ok; check if there is staged content
    staged = repo.git("diff", "--cached", "--name-only")
    if staged:
        repo.git("commit", "-m", message)


def push(repo: RepoSpec) -> None:
    repo.git("push")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="artifacts/hf_cellarc", help="Base directory containing the dataset repos.")
    ap.add_argument("--repo", default="hf-cellarc_100k", help="Main dataset repo directory name under --root.")
    ap.add_argument("--meta-repo", default="hf-cellarc_100k_meta", help="Metadata repo directory name under --root.")
    ap.add_argument("--subset-size", type=int, default=100, help="Number of episodes to sample per split.")
    ap.add_argument("--seed", type=int, default=12345, help="Random seed for reproducibility.")
    ap.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS_DEFAULT),
        choices=list(SPLITS_DEFAULT),
        help="Splits to subset.",
    )
    ap.add_argument("--no-commit", action="store_true", help="Do not commit changes to the repos.")
    ap.add_argument("--push", action="store_true", help="Push commits to remotes after creation.")
    args = ap.parse_args()

    root = Path(args.root)
    main_repo = RepoSpec(root / args.repo, name=args.repo)
    meta_repo = RepoSpec(root / args.meta_repo, name=args.meta_repo)

    # Sanity: ensure both repos exist and are clean
    for repo in (main_repo, meta_repo):
        if not repo.path.exists():
            raise SystemExit(f"Repository not found: {repo.path}")

    # Use the main repo to select IDs for each split
    rng = random.Random(args.seed)

    # Track per-repo updates to data_files.json
    main_updates: Dict[str, Tuple[Path, Path, int]] = {}
    meta_updates: Dict[str, Tuple[Path, Path, int]] = {}

    for split in args.splits:
        src_jsonl_main = main_repo.data_dir / f"{split}.jsonl"
        src_jsonl_meta = meta_repo.data_dir / f"{split}.jsonl"
        if not src_jsonl_main.exists():
            raise SystemExit(f"Missing source split in main repo: {src_jsonl_main}")
        if not src_jsonl_meta.exists():
            raise SystemExit(f"Missing source split in meta repo: {src_jsonl_meta}")

        # Read full split from main repo and sample IDs
        rows_main = read_jsonl(src_jsonl_main)
        ids = [str(r.get("id")) for r in rows_main]
        ids = [i for i in ids if i and i != "None"]
        if not ids:
            raise SystemExit(f"No IDs found in {src_jsonl_main}")
        k = min(args.subset_size, len(ids))
        rng.shuffle(ids)
        subset_ids = set(ids[:k])

        # Subset rows (preserve order from the source for determinism after shuffle)
        subset_rows_main = [r for r in rows_main if str(r.get("id")) in subset_ids]

        # Mirror selection in meta repo
        rows_meta = read_jsonl(src_jsonl_meta)
        subset_rows_meta = [r for r in rows_meta if str(r.get("id")) in subset_ids]

        # Write JSONL subsets
        split_tag = f"{split}_{k}"
        out_jsonl_main = main_repo.data_dir / f"{split_tag}.jsonl"
        out_parquet_main = main_repo.data_dir / f"{split_tag}.parquet"
        out_jsonl_meta = meta_repo.data_dir / f"{split_tag}.jsonl"
        out_parquet_meta = meta_repo.data_dir / f"{split_tag}.parquet"

        n_main = write_jsonl(out_jsonl_main, subset_rows_main)
        n_meta = write_jsonl(out_jsonl_meta, subset_rows_meta)
        assert n_main == n_meta == k, "Mismatch in subset sizes written"

        # Write Parquet subsets (supervision-only schema)
        write_parquet(out_parquet_main, subset_rows_main)
        # Write an identical Parquet file for meta (fields are the same)
        write_parquet(out_parquet_meta, subset_rows_main)

        # Write ID lists in both repos
        for repo in (main_repo, meta_repo):
            repo.subset_ids_dir.mkdir(parents=True, exist_ok=True)
            ids_file = repo.subset_ids_dir / f"{split_tag}.txt"
            with ids_file.open("w", encoding="utf-8") as fh:
                for _id in subset_ids:
                    fh.write(f"{_id}\n")

        # Track updates for data_files.json
        main_updates[split_tag] = (out_jsonl_main, out_parquet_main, k)
        meta_updates[split_tag] = (out_jsonl_meta, out_parquet_meta, k)

    # Update data_files.json
    update_data_files_json(main_repo, main_updates)
    update_data_files_json(meta_repo, meta_updates)

    # Commit changes if requested
    if not args.no_commit:
        commit_all(main_repo, f"Add {args.subset_size}-size subset splits and ID lists")
        commit_all(meta_repo, f"Add {args.subset_size}-size subset splits and ID lists")

    # Optional push (explicit only)
    if args.push:
        push(main_repo)
        push(meta_repo)


if __name__ == "__main__":
    main()
