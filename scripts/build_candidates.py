from pathlib import Path
import argparse

import pandas as pd
from pyfaidx import Fasta

from splice_paths.sequence_and_candidates import (
    orient_gene_region,
    build_candidate_table_from_oriented_seq,
)


def write_candidate_tables(
    genome,
    genes_df: pd.DataFrame,
    out_path,
) -> int:
    """
    Build candidate table for all genes and write incrementally to disk.

    Returns
    -------
    total_rows:
        Number of candidate rows written.
    """

    first_write = True
    total_rows = 0

    for i, gene in genes_df.iterrows():
        gene_id = gene["gene_id"]
        seqid = gene["seqid"]
        strand = gene["strand"]
        gene_start = int(gene["start"])
        gene_end = int(gene["end"])

        try:
            seq = orient_gene_region(
                genome=genome,
                seqid=seqid,
                gene_start=gene_start,
                gene_end=gene_end,
                strand=strand,
            )
        except KeyError:
            print(f"Skipping gene {gene_id}: seqid {seqid} not found in FASTA")
            continue

        gene_candidates = build_candidate_table_from_oriented_seq(
            seq=seq,
            gene_id=gene_id,
            seqid=seqid,
            strand=strand,
        )

        if len(gene_candidates) == 0:
            continue

        gene_candidates.to_csv(
            out_path,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )

        first_write = False
        total_rows += len(gene_candidates)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} genes; written {total_rows} candidates")

    return total_rows

def main():
    parser = argparse.ArgumentParser(
        description="Build GT/AG candidate splice-site table from gene intervals."
    )

    parser.add_argument("--fasta", required=True, help="Path to genome FASTA")
    parser.add_argument("--genes", required=True, help="Path to genes.tsv")
    parser.add_argument("--out", required=True, help="Output candidates.tsv")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of genes to process for testing",
    )

    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading genes table")
    genes_df = pd.read_csv(args.genes, sep="\t")

    if args.limit is not None:
        genes_df = genes_df.head(args.limit).copy()
        print(f"Using first {args.limit} genes")

    print("Loading FASTA")
    genome = Fasta(args.fasta)

    print("Building candidate table")
    total_rows = write_candidate_tables(
        genome=genome,
        genes_df=genes_df,
        out_path=out_path,
    )
    
    print(f"Saved candidates to: {out_path}")
    print(f"Candidate rows: {total_rows}")

if __name__ == "__main__":
    main()
    