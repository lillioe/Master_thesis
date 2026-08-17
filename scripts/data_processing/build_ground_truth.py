from pathlib import Path
import argparse

import pandas as pd


GROUND_TRUTH_COLUMNS = [
    "gene_id",
    "candidate_index",
    "candidate_type",
    "type_index",
    "label",
    "is_used",
    "transcript_ids",
    "n_transcripts_used",
]


def iter_candidate_gene_groups(candidates_path: str, chunksize: int):
    """
    Stream candidates.tsv gene-by-gene,
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
    Convert transcript exon coordinates to transcript-oriented relative coordinates
    Output intervals [rel_start, rel_end).
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

        else:
            raise ValueError(f"Unexpected strand: {strand}")

        relative_exons.append((rel_start, rel_end))

    return sorted(relative_exons)


def collect_used_candidates_for_gene(
    gene_candidates: pd.DataFrame,
    gene_id: str,
    gene_start: int,
    gene_end: int,
    strand: str,
    gene_transcripts: pd.DataFrame,
    exons_by_tx: dict,
) -> pd.DataFrame:
    """
    Build aground truth for one gene,
    One output row per used candidate,
    transcript_ids contains all transcripts in which that candidate is used
    """

    candidate_lookup = {}

    for _, row in gene_candidates.iterrows():
        key = (int(row["relative_base_pos"]), row["candidate_type"])

        candidate_lookup[key] = {
            "candidate_index": int(row["candidate_index"]),
            "candidate_type": row["candidate_type"],
            "type_index": int(row["type_index"]) if "type_index" in row and pd.notna(row["type_index"]) else None,
        }

    used = {}

    for _, tx in gene_transcripts.iterrows():
        transcript_id = tx["transcript_id"]

        tx_exons = exons_by_tx.get((gene_id, transcript_id))

        if tx_exons is None or len(tx_exons) < 2:
            continue

        relative_exons = get_relative_exons(
            tx_exons_df=tx_exons,
            gene_start=gene_start,
            gene_end=gene_end,
            strand=strand,
        )

        for left_exon, right_exon in zip(relative_exons[:-1], relative_exons[1:]):
            _, left_end = left_exon
            right_start, _ = right_exon

            donor_pos = left_end
            acceptor_pos = right_start - 2

            for pos, candidate_type, label in [
                (donor_pos, "donor", "used_donor"),
                (acceptor_pos, "acceptor", "used_acceptor"),
            ]:
                candidate = candidate_lookup.get((pos, candidate_type))

                if candidate is None:
                    # Usually noncanonical splice site or coordinate issue
                    # Candidate table currently only contains GT/AG
                    continue

                candidate_index = candidate["candidate_index"]

                if candidate_index not in used:
                    used[candidate_index] = {
                        "gene_id": gene_id,
                        "candidate_index": candidate_index,
                        "candidate_type": candidate["candidate_type"],
                        "type_index": candidate["type_index"],
                        "label": label,
                        "is_used": 1,
                        "transcript_ids": set(),
                    }

                used[candidate_index]["transcript_ids"].add(transcript_id)

    rows = []

    for candidate_index in sorted(used):
        row = used[candidate_index]
        transcript_ids = sorted(row["transcript_ids"])

        rows.append({
            "gene_id": row["gene_id"],
            "candidate_index": row["candidate_index"],
            "candidate_type": row["candidate_type"],
            "type_index": row["type_index"],
            "label": row["label"],
            "is_used": row["is_used"],
            "transcript_ids": ";".join(transcript_ids),
            "n_transcripts_used": len(transcript_ids),
        })

    return pd.DataFrame(rows, columns=GROUND_TRUTH_COLUMNS)


def write_ground_truth_table(
    candidates_path: str,
    genes_df: pd.DataFrame,
    transcripts_df: pd.DataFrame,
    exons_df: pd.DataFrame,
    out_path: Path,
    chunksize: int,
    limit_genes: int | None = None,
) -> int:
    """
    Stream candidates, write ground truth.
    """

    first_write = True
    total_rows = 0
    total_genes = 0

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
            print(f"Skipping gene {gene_id}: not found in genes table")
            continue

        gene_info = genes_by_id.loc[gene_id]

        if isinstance(gene_info, pd.DataFrame):
            gene_info = gene_info.iloc[0]

        gene_start = int(gene_info["start"])
        gene_end = int(gene_info["end"])
        strand = gene_info["strand"]

        gene_transcripts = transcripts_by_gene.get(gene_id)

        if gene_transcripts is None:
            total_genes += 1
            continue

        gt_gene = collect_used_candidates_for_gene(
            gene_candidates=gene_candidates,
            gene_id=gene_id,
            gene_start=gene_start,
            gene_end=gene_end,
            strand=strand,
            gene_transcripts=gene_transcripts,
            exons_by_tx=exons_by_tx,
        )

        if len(gt_gene) > 0:
            gt_gene.to_csv(
                out_path,
                sep="\t",
                index=False,
                mode="w" if first_write else "a",
                header=first_write,
            )

            first_write = False
            total_rows += len(gt_gene)

        total_genes += 1

        if total_genes % 1000 == 0:
            print(
                f"Processed {total_genes} genes; "
                f"written {total_rows} used-candidate rows"
            )

    if first_write:
        pd.DataFrame(columns=GROUND_TRUTH_COLUMNS).to_csv(
            out_path,
            sep="\t",
            index=False,
        )

    return total_rows


def main():
    parser = argparse.ArgumentParser(
        description="Build ground truth for candidate splice sites"
    )

    parser.add_argument("--candidates", required=True, help="Path to candidates.tsv")
    parser.add_argument("--genes", required=True, help="Path to genes.tsv")
    parser.add_argument("--transcripts", required=True, help="Path to transcripts.tsv")
    parser.add_argument("--exons", required=True, help="Path to exons.tsv")
    parser.add_argument("--out", required=True, help="Output ground_truth_used_candidates.tsv")

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
        help="Optional number of genes to process for testing",
    )

    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading annotation tables")
    genes_df = pd.read_csv(args.genes, sep="\t")
    transcripts_df = pd.read_csv(args.transcripts, sep="\t")
    exons_df = pd.read_csv(args.exons, sep="\t")

    print("Building aggregated sparse ground truth")
    total_rows = write_ground_truth_table(
        candidates_path=args.candidates,
        genes_df=genes_df,
        transcripts_df=transcripts_df,
        exons_df=exons_df,
        out_path=out_path,
        chunksize=args.chunksize,
        limit_genes=args.limit_genes,
    )

    print(f"Saved ground truth to: {out_path}")
    print(f"Used-candidate rows: {total_rows}")


if __name__ == "__main__":
    main()