import pandas as pd
from Bio.Seq import Seq


def create_empty_tables():
    candidate_cols = [
        "gene_id",
        "seqid",
        "strand",
        "candidate_index",
        "motif",
        "candidate_type",
        "relative_base_pos",
        "relative_fraction",
    ]

    ground_truth_cols = [
        "gene_id",
        "transcript_id",
        "candidate_index",
        "label",
        "is_used",
    ]

    candidates_df = pd.DataFrame(columns=candidate_cols)
    ground_truth_df = pd.DataFrame(columns=ground_truth_cols)

    return candidates_df, ground_truth_df


def reverse_complement(seq: str) -> str:
    return str(Seq(seq).reverse_complement())


def orient_gene_region(
    genome,
    seqid: str,
    gene_start: int,
    gene_end: int,
    strand: str,
) -> str:
    """
    Extract gene region from genome and orient it 5' -> 3'.

    gene_start/gene_end are GFF-style 1-based inclusive coordinates.
    The returned sequence is transcript-oriented.
    """

    if strand not in {"+", "-"}:
        raise ValueError(f"Unexpected strand: {strand}")

    # GFF: 1-based inclusive
    # Python/pyfaidx slicing: 0-based, end-exclusive
    seq = genome[seqid][gene_start - 1: gene_end].seq.upper()

    if strand == "-":
        seq = reverse_complement(seq)

    return seq

# canddidate extraction
def build_candidate_table_from_oriented_seq(
    seq: str,
    gene_id: str,
    seqid: str,
    strand: str,
) -> pd.DataFrame:
    """
    Build candidate donor/acceptor table from an already oriented sequence.

    seq must already be 5' -> 3' relative to the gene.
    """

    seq = seq.upper()
    gene_length = len(seq)

    rows = []
    candidate_index = 0

    for pos in range(gene_length - 1):
        motif = seq[pos:pos + 2]

        if motif == "GT":
            candidate_type = "donor"
        elif motif == "AG":
            candidate_type = "acceptor"
        else:
            continue

        rows.append({
            "gene_id": gene_id,
            "seqid": seqid,
            "strand": strand,
            "candidate_index": candidate_index,
            "motif": motif,
            "candidate_type": candidate_type,
            "relative_base_pos": pos,
            "relative_fraction": pos / gene_length,
        })

        candidate_index += 1

    return pd.DataFrame(rows)

# Ground truth extraction
def build_ground_truth_for_transcript(
    candidates_df: pd.DataFrame,
    tx_exons_df: pd.DataFrame,
    gene_id: str,
    transcript_id: str,
    gene_start: int,
    gene_end: int,
    strand: str,
) -> pd.DataFrame:
    """
    Build simple ground-truth labels for one transcript.

    Output:
        one row per candidate
        label = unused / used_donor / used_acceptor
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

    relative_exons = sorted(relative_exons)

    used_sites = set()

    for left_exon, right_exon in zip(relative_exons[:-1], relative_exons[1:]):
        _, left_end = left_exon
        right_start, _ = right_exon

        donor_pos = left_end
        acceptor_pos = right_start - 2

        used_sites.add((donor_pos, "donor"))
        used_sites.add((acceptor_pos, "acceptor"))

    rows = []

    for _, cand in candidates_df.iterrows():
        candidate_index = int(cand["candidate_index"])
        candidate_type = cand["candidate_type"]
        pos = int(cand["relative_base_pos"])

        is_used = int((pos, candidate_type) in used_sites)

        if is_used and candidate_type == "donor":
            label = "used_donor"
        elif is_used and candidate_type == "acceptor":
            label = "used_acceptor"
        else:
            label = "unused"

        rows.append({
            "gene_id": gene_id,
            "transcript_id": transcript_id,
            "candidate_index": candidate_index,
            "label": label,
            "is_used": is_used,
        })

    return pd.DataFrame(rows)
