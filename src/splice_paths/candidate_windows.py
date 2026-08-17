from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from pyfaidx import Fasta


DNA_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    return seq.translate(DNA_COMP)[::-1].upper()


def normalize_seqid(seqid: str, genome: Fasta) -> str:
    """
    Handle chr1 vs 1 naming differences between candidate table and FASTA
    """
    seqid = str(seqid)

    if seqid in genome:
        return seqid

    if seqid.startswith("chr") and seqid[3:] in genome:
        return seqid[3:]

    with_chr = "chr" + seqid
    if with_chr in genome:
        return with_chr

    raise KeyError(f"Sequence ID {seqid!r} not found in FASTA.")


def oriented_pos_to_genomic_pos_1based(
    oriented_pos: int,
    gene_start: int,
    gene_end: int,
    strand: str,
    oriented_pos_system: str = "0based",
) -> int:
    """
    Convert a candidate position inside an oriented gene sequence back to
    a 1-based genomic coordinate

    Assumes gene_start and gene_end are 1-based inclusive coordinates
    """

    if oriented_pos_system == "0based":
        offset0 = int(oriented_pos)
    elif oriented_pos_system == "1based":
        offset0 = int(oriented_pos) - 1
    else:
        raise ValueError("oriented_pos_system must be '0based' or '1based'.")

    if strand == "+":
        return int(gene_start) + offset0

    if strand == "-":
        return int(gene_end) - offset0

    raise ValueError(f"Unexpected strand: {strand!r}")


def extract_oriented_window(
    genome: Fasta,
    seqid: str,
    genomic_pos: int,
    strand: str,
    upstream: int = 300,
    downstream: int = 300,
    coordinate_system: str = "1based",
    pad_with_n: bool = True,
) -> str:
    """
    Extract a fixed window around a candidate position.

    The returned sequence is in transcript orientation:
      + strand: genomic sequence as-is
      - strand: reverse-complemented sequence

    With upstream=300 and downstream=300, the output length is 600 bp,
    candidate anchor base is at index 300,
    total length divisible by 6 for NT
    """

    window_size = upstream + downstream

    if window_size % 6 != 0:
        raise ValueError(
            f"Window size must be divisible by 6 for NT. Got {window_size}."
        )

    seqid = normalize_seqid(seqid, genome)
    chrom_len = len(genome[seqid])

    if coordinate_system == "1based":
        center0 = int(genomic_pos) - 1
    elif coordinate_system == "0based":
        center0 = int(genomic_pos)
    else:
        raise ValueError("coordinate_system must be '1based' or '0based'.")

    if strand == "+":
        start0 = center0 - upstream
        end0 = center0 + downstream

    elif strand == "-":
        # After reverse-complementing, the candidate anchor is still
        # located at index `upstream`.
        start0 = center0 - downstream + 1
        end0 = center0 + upstream + 1

    else:
        raise ValueError(f"Unexpected strand: {strand!r}")

    left_pad = max(0, -start0)
    right_pad = max(0, end0 - chrom_len)

    fetch_start = max(0, start0)
    fetch_end = min(chrom_len, end0)

    seq = genome[seqid][fetch_start:fetch_end].seq.upper()

    if pad_with_n:
        seq = ("N" * left_pad) + seq + ("N" * right_pad)
    elif left_pad or right_pad:
        raise ValueError(
            f"Window outside chromosome bounds: "
            f"{seqid}:{genomic_pos} strand={strand}"
        )

    if len(seq) != window_size:
        raise RuntimeError(f"Expected {window_size} bp, got {len(seq)} bp.")

    if strand == "-":
        seq = reverse_complement(seq)

    return seq


def add_genomic_positions_from_oriented_gene_offsets(
    candidates: pd.DataFrame,
    genes: pd.DataFrame,
    oriented_pos_col: str,
    oriented_pos_system: str = "0based",
    gene_id_col: str = "gene_id",
    gene_start_col: str = "start",
    gene_end_col: str = "end",
    strand_col: str = "strand",
    output_col: str = "genomic_pos_1based",
) -> pd.DataFrame:
    """
    Add genomic coordinates to a candidate table when candidates are stored
    as offsets inside oriented gene sequences
    """

    required_candidate_cols = {gene_id_col, oriented_pos_col, strand_col}
    missing_candidate_cols = required_candidate_cols - set(candidates.columns)
    if missing_candidate_cols:
        raise ValueError(
            f"Candidate table missing columns: {missing_candidate_cols}"
        )

    required_gene_cols = {gene_id_col, gene_start_col, gene_end_col}
    missing_gene_cols = required_gene_cols - set(genes.columns)
    if missing_gene_cols:
        raise ValueError(f"Genes table missing columns: {missing_gene_cols}")

    df = candidates.merge(
        genes[[gene_id_col, gene_start_col, gene_end_col]],
        on=gene_id_col,
        how="left",
        validate="many_to_one",
    )

    if df[gene_start_col].isna().any() or df[gene_end_col].isna().any():
        missing = df.loc[df[gene_start_col].isna(), gene_id_col].unique()[:10]
        raise ValueError(
            f"Candidates could not be matched to genes. Examples: {missing}"
        )

    df[output_col] = [
        oriented_pos_to_genomic_pos_1based(
            oriented_pos=row[oriented_pos_col],
            gene_start=row[gene_start_col],
            gene_end=row[gene_end_col],
            strand=row[strand_col],
            oriented_pos_system=oriented_pos_system,
        )
        for _, row in df.iterrows()
    ]

    return df


def add_windows_to_candidates(
    candidates: pd.DataFrame,
    genome: Fasta,
    seqid_col: str = "seqid",
    strand_col: str = "strand",
    genomic_pos_col: str = "genomic_pos_1based",
    genomic_pos_system: str = "1based",
    upstream: int = 300,
    downstream: int = 300,
    window_col: Optional[str] = None,
    add_qc_cols: bool = True,
) -> pd.DataFrame:
    """
    Return a copy of the candidate table with an added DNA window column
    """

    required_cols = {seqid_col, strand_col, genomic_pos_col}
    missing_cols = required_cols - set(candidates.columns)
    if missing_cols:
        raise ValueError(f"Candidate table missing columns: {missing_cols}")

    df = candidates.copy()
    window_size = upstream + downstream

    if window_col is None:
        window_col = f"window_seq_{window_size}bp"

    windows = []

    for i, row in df.iterrows():
        try:
            seq = extract_oriented_window(
                genome=genome,
                seqid=row[seqid_col],
                genomic_pos=row[genomic_pos_col],
                strand=row[strand_col],
                upstream=upstream,
                downstream=downstream,
                coordinate_system=genomic_pos_system,
                pad_with_n=True,
            )
        windows.append(seq)

    df[window_col] = windows

    if add_qc_cols:
        center = upstream
        df[f"center_index_{window_size}bp"] = center
        df[f"center_base_{window_size}bp"] = [seq[center] for seq in windows]
        df[f"center_2bp_{window_size}bp"] = [
            seq[center : center + 2] for seq in windows
        ]
        df[f"center_6bp_{window_size}bp"] = [
            seq[center - 3 : center + 3] for seq in windows
        ]

    return df


def append_windows_to_candidate_file(
    fasta_path: str | Path,
    candidates_path: str | Path,
    out_path: str | Path,
    seqid_col: str = "seqid",
    strand_col: str = "strand",
    genomic_pos_col: str = "genomic_pos_1based",
    genomic_pos_system: str = "1based",
    upstream: int = 300,
    downstream: int = 300,
) -> int:
    """
    File wrapper, reads candidates.tsv, 
    appends window sequence column, writes output TSV.
    """

    genome = Fasta(str(fasta_path))
    candidates = pd.read_csv(candidates_path, sep="\t")

    out = add_windows_to_candidates(
        candidates=candidates,
        genome=genome,
        seqid_col=seqid_col,
        strand_col=strand_col,
        genomic_pos_col=genomic_pos_col,
        genomic_pos_system=genomic_pos_system,
        upstream=upstream,
        downstream=downstream,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(out_path, sep="\t", index=False)

    return len(out)