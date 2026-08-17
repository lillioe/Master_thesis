from pathlib import Path
import argparse
import re
import pandas as pd


LABEL_3_TO_ID = {
    "skip": 0,
    "donor": 1,
    "acceptor": 2,
}

LABEL_4_TO_ID = {
    "S_D": 0,   # skip while waiting for next selected donor
    "D": 1,     # selected donor
    "S_A": 2,   # skip while waiting for next selected acceptor
    "A": 3,     # selected acceptor
}


BASE_COLUMNS = [
    "gene_id",
    "transcript_id",
    "seqid",
    "strand",
    "candidate_index",
    "relative_base_pos",
    "candidate_type",
    "type_index",
    "is_selected",
    "selected_order",
    "n_candidates_in_gene",
    "n_selected_candidates",
]


def get_output_columns(label_mode: str) -> list[str]:
    columns = BASE_COLUMNS.copy()
    insert_at = columns.index("is_selected")

    if label_mode in {"3", "both"}:
        columns[insert_at:insert_at] = ["label_3", "label_3_id"]
        insert_at += 2

    if label_mode in {"4", "both"}:
        columns[insert_at:insert_at] = ["label_4", "label_4_id"]

    return columns


def safe_filename(value: str) -> str:
    """
    Convert seqid values to tsv filenames
    """
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return f"seqid={value}.tsv"


def iter_candidate_gene_groups(candidates_path: str, chunksize: int):
    """
    Stream candidates.tsv by gene
    assumes ordered by gene
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


def parse_int_list(value) -> list[int]:
    """
    Parse integer lists
    """

    if pd.isna(value):
        return []

    value = str(value).strip()

    if value == "":
        return []

    return [int(x) for x in value.split(";") if x != ""]


def assign_labels_for_transcript(
    gene_candidates: pd.DataFrame,
    path_row: pd.Series,
    label_mode: str,
) -> pd.DataFrame:
    """
    Expand one transcript path to one row per candidate in the gene.

    3-state labels:
        skip / donor / acceptor

    4-state labels:
        S_D = skip while waiting for next selected donor
        D   = selected donor
        S_A = skip while waiting for next selected acceptor
        A   = selected acceptor
    """

    selected_indices = parse_int_list(path_row["selected_candidate_indices"])
    selected_index_set = set(selected_indices)

    selected_order = {
        candidate_index: i
        for i, candidate_index in enumerate(selected_indices)
    }

    gene_candidates = gene_candidates.sort_values("candidate_index").copy()

    n_candidates_in_gene = len(gene_candidates)
    n_selected_candidates = len(selected_indices)

    expected_next = "donor"
    rows = []

    for _, cand in gene_candidates.iterrows():
        candidate_index = int(cand["candidate_index"])
        candidate_type = cand["candidate_type"]

        is_selected = candidate_index in selected_index_set

        out = {
            "gene_id": path_row["gene_id"],
            "transcript_id": path_row["transcript_id"],
            "seqid": path_row["seqid"],
            "strand": path_row["strand"],
            "candidate_index": candidate_index,
            "relative_base_pos": int(cand["relative_base_pos"]),
            "candidate_type": candidate_type,
            "type_index": int(cand["type_index"]) if "type_index" in cand.index and pd.notna(cand["type_index"]) else -1,
            "is_selected": int(is_selected),
            "selected_order": selected_order[candidate_index] if is_selected else -1,
            "n_candidates_in_gene": n_candidates_in_gene,
            "n_selected_candidates": n_selected_candidates,
        }

        if is_selected:
            if candidate_type == "donor":
                label_3 = "donor"
                label_4 = "D"
                expected_next = "acceptor"

            elif candidate_type == "acceptor":
                label_3 = "acceptor"
                label_4 = "A"
                expected_next = "donor"

            else:
                raise ValueError(f"Unexpected candidate_type: {candidate_type}")

        else:
            label_3 = "skip"

            if expected_next == "donor":
                label_4 = "S_D"
            elif expected_next == "acceptor":
                label_4 = "S_A"
            else:
                raise ValueError(f"Unexpected expected_next: {expected_next}")

        if label_mode in {"3", "both"}:
            out["label_3"] = label_3
            out["label_3_id"] = LABEL_3_TO_ID[label_3]

        if label_mode in {"4", "both"}:
            out["label_4"] = label_4
            out["label_4_id"] = LABEL_4_TO_ID[label_4]

        rows.append(out)

    return pd.DataFrame(rows)


def check_selected_path_consistency(labels_df: pd.DataFrame) -> bool:
    """
    Check that selected candidates alternate donor/acceptor
    """

    selected = labels_df[labels_df["is_selected"] == 1].sort_values("candidate_index")

    if len(selected) == 0:
        return False

    observed = selected["candidate_type"].tolist()
    expected = ["donor" if i % 2 == 0 else "acceptor" for i in range(len(observed))]

    return observed == expected


def prepare_output_paths(
    out_path: str | None,
    out_dir: str | None,
    split_by_seqid: bool,
    overwrite: bool,
) -> tuple[Path | None, Path | None]:
    """
    Prepare output path or output directory
    """

    if split_by_seqid:
        if out_dir is None:
            raise ValueError("--out-dir is required when using --split-by-seqid")

        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        existing_tsvs = list(out_dir_path.glob("*.tsv"))
        existing_manifest = out_dir_path / "manifest.tsv"

        if existing_tsvs or existing_manifest.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output directory already contains TSV files: {out_dir_path}\n"
                    "Use --overwrite to remove existing TSV outputs."
                )

            for path in existing_tsvs:
                path.unlink()

            if existing_manifest.exists():
                existing_manifest.unlink()

        return None, out_dir_path

    if out_path is None:
        raise ValueError("--out is required unless using --split-by-seqid")

    out_file_path = Path(out_path)
    out_file_path.parent.mkdir(parents=True, exist_ok=True)

    if out_file_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {out_file_path}\n"
            "Use --overwrite to replace it."
        )

    if out_file_path.exists() and overwrite:
        out_file_path.unlink()

    return out_file_path, None


def write_gene_output(
    gene_out: pd.DataFrame,
    output_columns: list[str],
    out_file_path: Path | None,
    out_dir_path: Path | None,
    split_by_seqid: bool,
    written_files: set[Path],
    rows_by_file: dict[Path, int],
) -> None:
    """
    Write one gene's expanded labels either to one combined TSV
    or to a per-seqid TSV.
    """

    gene_out = gene_out[output_columns]

    if split_by_seqid:
        seqids = gene_out["seqid"].dropna().unique()

        seqid = str(seqids[0])
        out_path = out_dir_path / safe_filename(seqid)

    else:
        out_path = out_file_path

    first_write = out_path not in written_files

    gene_out.to_csv(
        out_path,
        sep="\t",
        index=False,
        mode="w" if first_write else "a",
        header=first_write,
    )

    written_files.add(out_path)
    rows_by_file[out_path] = rows_by_file.get(out_path, 0) + len(gene_out)


def write_manifest(out_dir_path: Path, rows_by_file: dict[Path, int]) -> None:
    """
    Write small manifest to show which files created, how many rows
    """

    rows = []

    for path, n_rows in sorted(rows_by_file.items(), key=lambda x: str(x[0])):
        filename = path.name

        if filename.startswith("seqid=") and filename.endswith(".tsv"):
            seqid = filename[len("seqid="):-len(".tsv")]
        else:
            seqid = ""

        rows.append({
            "seqid": seqid,
            "file": str(path),
            "n_rows": n_rows,
        })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir_path / "manifest.tsv", sep="\t", index=False)


def write_transcript_candidate_labels(
    candidates_path: str,
    transcript_paths_df: pd.DataFrame,
    out_file_path: Path | None,
    out_dir_path: Path | None,
    split_by_seqid: bool,
    label_mode: str,
    chunksize: int,
    limit_genes: int | None = None,
    limit_transcripts: int | None = None,
) -> int:
    """
    Build expanded per candidate labels for each valid transcript path
    """

    output_columns = get_output_columns(label_mode)

    total_rows = 0
    total_genes = 0
    total_transcripts = 0
    inconsistent_paths = 0

    written_files = set()
    rows_by_file = {}

    if "path_valid" in transcript_paths_df.columns:
        transcript_paths_df = transcript_paths_df[
            transcript_paths_df["path_valid"] == 1
        ].copy()

    if limit_transcripts is not None:
        transcript_paths_df = transcript_paths_df.head(limit_transcripts).copy()

    paths_by_gene = {
        gene_id: df
        for gene_id, df in transcript_paths_df.groupby("gene_id", sort=False)
    }

    genes_needed = set(paths_by_gene)

    for gene_id, gene_candidates in iter_candidate_gene_groups(
        candidates_path=candidates_path,
        chunksize=chunksize,
    ):
        if gene_id not in genes_needed:
            continue

        if limit_genes is not None and total_genes >= limit_genes:
            break

        gene_paths = paths_by_gene[gene_id]
        gene_chunks = []

        for _, path_row in gene_paths.iterrows():
            labels_df = assign_labels_for_transcript(
                gene_candidates=gene_candidates,
                path_row=path_row,
                label_mode=label_mode,
            )

            if not check_selected_path_consistency(labels_df):
                inconsistent_paths += 1
                print(
                    "Warning: selected path does not alternate donor/acceptor for "
                    f"{path_row['gene_id']} {path_row['transcript_id']}"
                )

            gene_chunks.append(labels_df)
            total_transcripts += 1

        if gene_chunks:
            gene_out = pd.concat(gene_chunks, ignore_index=True)

            write_gene_output(
                gene_out=gene_out,
                output_columns=output_columns,
                out_file_path=out_file_path,
                out_dir_path=out_dir_path,
                split_by_seqid=split_by_seqid,
                written_files=written_files,
                rows_by_file=rows_by_file,
            )

            total_rows += len(gene_out)

        total_genes += 1

        if total_genes % 1000 == 0:
            print(
                f"Processed {total_genes} genes with transcript paths; "
                f"{total_transcripts} transcripts; "
                f"{total_rows} rows written"
            )

    if split_by_seqid:
        write_manifest(out_dir_path, rows_by_file)
    else:
        if out_file_path not in written_files:
            pd.DataFrame(columns=output_columns).to_csv(
                out_file_path,
                sep="\t",
                index=False,
            )

    print(f"Genes processed: {total_genes}")
    print(f"Transcript paths expanded: {total_transcripts}")
    print(f"Inconsistent selected paths: {inconsistent_paths}")
    print(f"Output files written: {len(written_files)}")

    return total_rows


def main():
    parser = argparse.ArgumentParser(
        description="Expand valid transcript paths into per-candidate CRF labels"
    )

    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates.tsv",
    )

    parser.add_argument(
        "--transcript-paths",
        required=True,
        help="Path to transcript_paths_valid.tsv or transcript_paths.tsv",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Output TSV path for one combined file. Not used with --split-by-seqid",
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for per seqid TSV files. Required with --split-by-seqid",
    )

    parser.add_argument(
        "--split-by-seqid",
        action="store_true",
        help="Write one TSV file per seqid/chromosome instead of one combined TSV",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output TSVs.",
    )

    parser.add_argument(
        "--label-mode",
        choices=["3", "4", "both"],
        default="both",
        help="Which labels to write: 3-state, 4-state, or both",
    )

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
        help="Optional number of genes with transcript paths to process for testing",
    )

    parser.add_argument(
        "--limit-transcripts",
        type=int,
        default=None,
        help="Optional number of transcript paths to process for testing",
    )

    args = parser.parse_args()

    out_file_path, out_dir_path = prepare_output_paths(
        out_path=args.out,
        out_dir=args.out_dir,
        split_by_seqid=args.split_by_seqid,
        overwrite=args.overwrite,
    )

    print("Loading transcript paths")
    transcript_paths_df = pd.read_csv(args.transcript_paths, sep="\t")

    print("Building transcript-candidate CRF labels")
    print(f"Label mode: {args.label_mode}")
    print(f"Split by seqid: {args.split_by_seqid}")

    total_rows = write_transcript_candidate_labels(
        candidates_path=args.candidates,
        transcript_paths_df=transcript_paths_df,
        out_file_path=out_file_path,
        out_dir_path=out_dir_path,
        split_by_seqid=args.split_by_seqid,
        label_mode=args.label_mode,
        chunksize=args.chunksize,
        limit_genes=args.limit_genes,
        limit_transcripts=args.limit_transcripts,
    )

    if args.split_by_seqid:
        print(f"Saved per-seqid transcript-candidate labels to: {out_dir_path}")
        print(f"Manifest: {out_dir_path / 'manifest.tsv'}")
    else:
        print(f"Saved transcript-candidate labels to: {out_file_path}")

    print(f"Rows written: {total_rows}")


if __name__ == "__main__":
    main()
