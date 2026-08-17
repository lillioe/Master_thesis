from pathlib import Path
import argparse

import pandas as pd
from pyfaidx import Fasta

from src.splice_paths.candidate_windows import (
    add_genomic_positions_from_oriented_gene_offsets,
    add_windows_to_candidates,
)


def main():
    parser = argparse.ArgumentParser(
        description="Append DNA windows to a candidate table in chunks"
    )

    parser.add_argument("--fasta", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--genes", default=None)
    parser.add_argument("--gene-id-col", default="gene_id")
    parser.add_argument("--seqid-col", default="seqid")
    parser.add_argument("--strand-col", default="strand")

    parser.add_argument("--genomic-pos-col", default=None)
    parser.add_argument("--genomic-pos-system", choices=["0based", "1based"], default="1based")

    parser.add_argument("--oriented-pos-col", default=None)
    parser.add_argument("--oriented-pos-system", choices=["0based", "1based"], default="0based")

    parser.add_argument("--upstream", type=int, default=300)
    parser.add_argument("--downstream", type=int, default=300)
    parser.add_argument("--chunksize", type=int, default=100_000)

    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Optional split filter, e.g. --splits train val",
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional limit for testing.",
    )

    args = parser.parse_args()

    window_size = args.upstream + args.downstream
    if window_size % 6 != 0:
        raise ValueError(f"Window size must be divisible by 6. Got {window_size}.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading genome FASTA")
    genome = Fasta(args.fasta)

    genes = None
    if args.genomic_pos_col is None:
        if args.genes is None or args.oriented_pos_col is None:
            raise ValueError(
                "Either provide --genomic-pos-col, or provide both "
                "--genes and --oriented-pos-col."
            )

        print("Loading genes table")
        genes = pd.read_csv(args.genes, sep="\t")

    first_write = True
    total_written = 0

    reader = pd.read_csv(
        args.candidates,
        sep="\t",
        chunksize=args.chunksize,
    )

    for chunk_i, candidates in enumerate(reader, start=1):
        if args.max_chunks is not None and chunk_i > args.max_chunks:
            break

        print(f"Processing chunk {chunk_i}: {len(candidates):,} rows")

        if args.splits is not None:
            candidates = candidates[candidates["split"].isin(args.splits)].copy()
            print(f"  after split filter: {len(candidates):,} rows")

        if len(candidates) == 0:
            continue

        if args.genomic_pos_col is None:
            candidates = add_genomic_positions_from_oriented_gene_offsets(
                candidates=candidates,
                genes=genes,
                oriented_pos_col=args.oriented_pos_col,
                oriented_pos_system=args.oriented_pos_system,
                gene_id_col=args.gene_id_col,
                strand_col=args.strand_col,
                output_col="genomic_pos_1based",
            )
            genomic_pos_col = "genomic_pos_1based"
            genomic_pos_system = "1based"
        else:
            genomic_pos_col = args.genomic_pos_col
            genomic_pos_system = args.genomic_pos_system

        candidates_with_windows = add_windows_to_candidates(
            candidates=candidates,
            genome=genome,
            seqid_col=args.seqid_col,
            strand_col=args.strand_col,
            genomic_pos_col=genomic_pos_col,
            genomic_pos_system=genomic_pos_system,
            upstream=args.upstream,
            downstream=args.downstream,
            add_qc_cols=True,
        )

        candidates_with_windows.to_csv(
            out_path,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )

        first_write = False
        total_written += len(candidates_with_windows)

        print(f"  total written: {total_written:,}")

    print(f"Saved: {out_path}")
    print(f"Total rows written: {total_written:,}")


if __name__ == "__main__":
    main()