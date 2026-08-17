from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--window-tsv", required=True)
    ap.add_argument("--transcript-paths-valid", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--n-train-genes", type=int, default=1000)
    ap.add_argument("--n-val-genes", type=int, default=200)
    ap.add_argument("--min-candidates", type=int, default=100)
    ap.add_argument("--max-candidates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--chunksize", type=int, default=500_000)

    args = ap.parse_args()

    window_tsv = Path(args.window_tsv)
    transcript_paths_valid = Path(args.transcript_paths_valid)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading valid transcript paths")
    paths = pd.read_csv(
        transcript_paths_valid,
        sep="\t",
        usecols=["gene_id", "transcript_id"],
    )

    valid_gene_summary = (
        paths.groupby("gene_id")
        .agg(n_valid_transcripts=("transcript_id", "nunique"))
        .reset_index()
    )

    valid_genes = set(valid_gene_summary["gene_id"])
    print("Valid genes:", len(valid_genes))

    print("Scanning window TSV")
    header = pd.read_csv(window_tsv, sep="\t", nrows=0).columns.tolist()

    required = ["gene_id", "seqid", "split", "candidate_index"]
    missing = [c for c in required if c not in header]
    gene_rows = []

    for chunk_i, chunk in enumerate(
        pd.read_csv(
            window_tsv,
            sep="\t",
            usecols=required,
            chunksize=args.chunksize,
        )
    ):
        chunk = chunk[chunk["gene_id"].isin(valid_genes)].copy()

        if len(chunk) == 0:
            continue

        summary = (
            chunk.groupby(["gene_id", "seqid", "split"], dropna=False)
            .agg(n_candidates=("candidate_index", "nunique"))
            .reset_index()
        )

        gene_rows.append(summary)

        print(f"chunk {chunk_i}; matching rows: {len(chunk):,}")


    gene_summary = pd.concat(gene_rows, ignore_index=True)

    gene_summary = (
        gene_summary.groupby(["gene_id", "seqid", "split"], dropna=False)
        .agg(n_candidates=("n_candidates", "sum"))
        .reset_index()
    )

    gene_summary = gene_summary.merge(valid_gene_summary, on="gene_id", how="left")

    gene_summary = gene_summary[
        (gene_summary["n_candidates"] >= args.min_candidates)
        & (gene_summary["n_candidates"] <= args.max_candidates)
    ].copy()

    print("\nEligible genes by split:")
    print(gene_summary.groupby("split")["gene_id"].nunique())

    rng = np.random.default_rng(args.seed)

    selected_parts = []

    split_requests = [
        ("train", args.n_train_genes),
        ("val", args.n_val_genes),
        ("validation", args.n_val_genes),
    ]

    for split, n in split_requests:
        sub = gene_summary[gene_summary["split"].astype(str) == split].copy()

        if len(sub) == 0:
            continue

        n_take = min(n, len(sub))
        chosen_idx = rng.choice(sub.index.to_numpy(), size=n_take, replace=False)
        selected_parts.append(sub.loc[chosen_idx].copy())


    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values(["split", "seqid", "gene_id"]).reset_index(drop=True)

    selected_genes = set(selected["gene_id"])

    selected_path = out_dir / "selected_crf_genes.tsv"
    selected.to_csv(selected_path, sep="\t", index=False)

    print("\nSaved:", selected_path)
    print(selected.groupby("split")["gene_id"].nunique())

    print("\n subset window TSV")
    subset_path = out_dir / "selected_crf_candidate_windows.tsv"

    first = True
    total_rows = 0

    for chunk_i, chunk in enumerate(
        pd.read_csv(window_tsv, sep="\t", chunksize=args.chunksize)
    ):
        sub = chunk[chunk["gene_id"].isin(selected_genes)].copy()

        if len(sub) == 0:
            continue

        sub.to_csv(
            subset_path,
            sep="\t",
            index=False,
            mode="w" if first else "a",
            header=first,
        )

        first = False
        total_rows += len(sub)

        print(f"chunk {chunk_i}; written rows: {total_rows:,}")

    print("Saved:", subset_path)
    print("Rows written:", total_rows)


if __name__ == "__main__":
    main()
