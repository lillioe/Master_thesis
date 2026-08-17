from pathlib import Path
import argparse
import pandas as pd


OUTPUT_COLUMNS = [
    "gene_id",
    "transcript_id",
    "seqid",
    "strand",
    "gene_start",
    "gene_end",
    "n_exons",
    "n_introns",
    "selected_candidate_indices",
    "selected_candidate_types",
    "selected_relative_positions",
    "n_selected_candidates",
    "n_missing_sites",
    "missing_sites",
    "path_valid",
]


def iter_candidate_gene_groups(candidates_path: str, chunksize: int):
    """
    Stream candidates.tsv gene-by-gene
    Assumes candidates.tsv is ordered by gene_id
    """

    buffer = pd.DataFrame()

    reader = pd.read_csv(
        candidates_path,
        sep="\t",
        chunksize=chunksize,
    )

    for chunk in reader:
        if not buffer.empty:
            chunk = pd.concat([buffer, chunk], ignore_index=True)

        last_gene_id = chunk["gene_id"].iloc[-1]

        complete = chunk[chunk["gene_id"] != last_gene_id]
        buffer = chunk[chunk["gene_id"] == last_gene_id].copy()

        if len(complete) > 0:
            for gene_id, gene_candidates in complete.groupby("gene_id", sort=False):
                yield gene_id, gene_candidates

    if not buffer.empty:
        for gene_id, gene_candidates in buffer.groupby("gene_id", sort=False):
            yield gene_id, gene_candidates


def get_relative_exons(
    tx_exons_df: pd.DataFrame,
    gene_start: int,
    gene_end: int,
    strand: str,
) -> list[tuple[int, int]]:
    """
    Convert transcript exon coordinates to relative coords
    Input exon coords are GFF-style 1-based 
    Output intervals: [rel_start, rel_end).
    """

    relative_exons = []

    for _, exon in tx_exons_df.iterrows():
        exon_start = int(exon["start"])
        exon_end = int(exon["end"])

        if strand == "+":
            rel_start = exon_start - gene_start
            rel_end = exon_end - gene_start + 1

        elif strand == "-":
            rel_start = gene_end - exon_end
            rel_end = gene_end - exon_start + 1

        relative_exons.append((rel_start, rel_end))

    return sorted(relative_exons)


def get_splice_sites_from_relative_exons(
    relative_exons: list[tuple[int, int]],
) -> list[tuple[int, str]]:
    """
    Return selected splice sites in transcript order.
    Each site is represented as:
        (relative_base_pos, candidate_type)
    Matches build_ground_truth.py:
        donor_pos = left_exon_end
        acceptor_pos = right_exon_start - 2
    """

    selected_sites = []

    if len(relative_exons) < 2:
        return selected_sites

    for left_exon, right_exon in zip(relative_exons[:-1], relative_exons[1:]):
        _, left_end = left_exon
        right_start, _ = right_exon

        donor_pos = left_end
        acceptor_pos = right_start - 2

        selected_sites.append((donor_pos, "donor"))
        selected_sites.append((acceptor_pos, "acceptor"))

    return selected_sites


def build_candidate_lookup(gene_candidates: pd.DataFrame) -> dict:
    """
    Map (relative_base_pos, candidate_type) to candidate information
    """

    lookup = {}

    for _, row in gene_candidates.iterrows():
        key = (int(row["relative_base_pos"]), row["candidate_type"])

        lookup[key] = {
            "candidate_index": int(row["candidate_index"]),
            "candidate_type": row["candidate_type"],
            "relative_base_pos": int(row["relative_base_pos"]),
        }

    return lookup


def build_transcript_path_row(
    gene_id: str,
    transcript_id: str,
    seqid: str,
    strand: str,
    gene_start: int,
    gene_end: int,
    tx_exons: pd.DataFrame,
    candidate_lookup: dict,
) -> dict:
    """
    Build one compact transcript path row
    """

    relative_exons = get_relative_exons(
        tx_exons_df=tx_exons,
        gene_start=gene_start,
        gene_end=gene_end,
        strand=strand,
    )

    n_exons = len(relative_exons)
    n_introns = max(0, n_exons - 1)

    selected_sites = get_splice_sites_from_relative_exons(relative_exons)

    selected_candidate_indices = []
    selected_candidate_types = []
    selected_relative_positions = []
    missing_sites = []

    for relative_pos, candidate_type in selected_sites:
        candidate = candidate_lookup.get((relative_pos, candidate_type))

        if candidate is None:
            missing_sites.append(f"{candidate_type}:{relative_pos}")
            continue

        selected_candidate_indices.append(str(candidate["candidate_index"]))
        selected_candidate_types.append(candidate_type)
        selected_relative_positions.append(str(relative_pos))

    n_selected_candidates = len(selected_candidate_indices)
    n_missing_sites = len(missing_sites)

    expected_selected = 2 * n_introns
    path_valid = int(
        n_introns > 0
        and n_missing_sites == 0
        and n_selected_candidates == expected_selected
    )

    return {
        "gene_id": gene_id,
        "transcript_id": transcript_id,
        "seqid": seqid,
        "strand": strand,
        "gene_start": gene_start,
        "gene_end": gene_end,
        "n_exons": n_exons,
        "n_introns": n_introns,
        "selected_candidate_indices": ";".join(selected_candidate_indices),
        "selected_candidate_types": ";".join(selected_candidate_types),
        "selected_relative_positions": ";".join(selected_relative_positions),
        "n_selected_candidates": n_selected_candidates,
        "n_missing_sites": n_missing_sites,
        "missing_sites": ";".join(missing_sites),
        "path_valid": path_valid,
    }


def write_transcript_paths(
    candidates_path: str,
    genes_df: pd.DataFrame,
    transcripts_df: pd.DataFrame,
    exons_df: pd.DataFrame,
    out_path: Path,
    chunksize: int,
    limit_genes: int | None = None,
) -> int:
    """
    Build compact transcript path table
    One row per transcript with at least two exons
    """

    first_write = True
    total_rows = 0
    total_genes = 0
    total_transcripts_seen = 0
    total_single_exon_skipped = 0

    genes_by_id = genes_df.set_index("gene_id", drop=False)

    transcripts_by_gene = {
        gene_id: df
        for gene_id, df in transcripts_df.groupby("gene_id", sort=False)
    }

    exons_by_tx = {
        key: df
        for key, df in exons_df.groupby(["gene_id", "transcript_id"], sort=False)
    }

    for gene_id, gene_candidates in iter_candidate_gene_groups(
        candidates_path=candidates_path,
        chunksize=chunksize,
    ):
        if limit_genes is not None and total_genes >= limit_genes:
            break

        if gene_id not in genes_by_id.index:
            print(f"Skipping gene {gene_id}")
            continue

        gene_info = genes_by_id.loc[gene_id]
        if isinstance(gene_info, pd.DataFrame):
            gene_info = gene_info.iloc[0]

        seqid = gene_info["seqid"]
        strand = gene_info["strand"]
        gene_start = int(gene_info["start"])
        gene_end = int(gene_info["end"])

        gene_transcripts = transcripts_by_gene.get(gene_id)
        if gene_transcripts is None:
            total_genes += 1
            continue

        candidate_lookup = build_candidate_lookup(gene_candidates)
        rows = []

        for _, tx in gene_transcripts.iterrows():
            transcript_id = tx["transcript_id"]
            total_transcripts_seen += 1

            tx_exons = exons_by_tx.get((gene_id, transcript_id))

            if tx_exons is None or len(tx_exons) < 2:
                total_single_exon_skipped += 1
                continue

            row = build_transcript_path_row(
                gene_id=gene_id,
                transcript_id=transcript_id,
                seqid=seqid,
                strand=strand,
                gene_start=gene_start,
                gene_end=gene_end,
                tx_exons=tx_exons,
                candidate_lookup=candidate_lookup,
            )

            rows.append(row)

        if rows:
            out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

            out_df.to_csv(
                out_path,
                sep="\t",
                index=False,
                mode="w" if first_write else "a",
                header=first_write,
            )

            first_write = False
            total_rows += len(out_df)

        total_genes += 1

        if total_genes % 1000 == 0:
            print(
                f"Processed {total_genes} genes; "
                f"written {total_rows} transcript paths"
            )

    if first_write:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(
            out_path,
            sep="\t",
            index=False,
        )

    print(f"Transcripts seen: {total_transcripts_seen}")
    print(f"Single-exon/no-exon transcripts skipped: {total_single_exon_skipped}")

    return total_rows


def main():
    parser = argparse.ArgumentParser(
        description="Build compact transcript-level splice paths for CRF training"
    )

    parser.add_argument("--candidates", required=True, help="Path to candidates.tsv")
    parser.add_argument("--genes", required=True, help="Path to genes.tsv")
    parser.add_argument("--transcripts", required=True, help="Path to transcripts.tsv")
    parser.add_argument("--exons", required=True, help="Path to exons.tsv")
    parser.add_argument("--out", required=True, help="Output transcript_paths.tsv")

    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Number of candidate rows to read per chunk",
    )

    parser.add_argument(
        "--limit-genes",
        type=int,
        default=None,
        help="number of genes to process for testing",
    )

    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading annotation tables")
    genes_df = pd.read_csv(args.genes, sep="\t")
    transcripts_df = pd.read_csv(args.transcripts, sep="\t")
    exons_df = pd.read_csv(args.exons, sep="\t")

    print("Building transcript paths")
    total_rows = write_transcript_paths(
        candidates_path=args.candidates,
        genes_df=genes_df,
        transcripts_df=transcripts_df,
        exons_df=exons_df,
        out_path=out_path,
        chunksize=args.chunksize,
        limit_genes=args.limit_genes,
    )

    print(f"Saved transcript paths to: {out_path}")
    print(f"Transcript path rows: {total_rows}")


if __name__ == "__main__":
    main()