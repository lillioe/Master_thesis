from pathlib import Path
import argparse

from splice_paths.annotation import (
    load_gff,
    build_genes_df,
    build_transcripts_df,
    build_exons_df,
    filter_to_multi_exon,
)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare genes, transcripts, and exons tables from RefSeq/NCBI GFF"
    )

    parser.add_argument("--gff", required=True, help="Path to genomic.gff.gz")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--multi-exon-only",
        action="store_true",
        help="Keep only transcripts with at least two exons",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gff = load_gff(args.gff)

    genes_df = build_genes_df(gff)
    transcripts_df = build_transcripts_df(gff)
    exons_df = build_exons_df(gff, transcripts_df)

    if args.multi_exon_only:
        genes_df, transcripts_df, exons_df = filter_to_multi_exon(
            genes_df,
            transcripts_df,
            exons_df,
        )

    genes_df.to_csv(out_dir / "genes.tsv", sep="\t", index=False)
    transcripts_df.to_csv(out_dir / "transcripts.tsv", sep="\t", index=False)
    exons_df.to_csv(out_dir / "exons.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()