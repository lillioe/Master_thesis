#!/usr/bin/env python3
"""
Validation selected clustering thresholds to the four CRF test posterior
sample files.

Script loads frozen val thresholds, processes four test posterior files, 
saves model, resumes from exisitng model output, writes summary

Run against the mounted ERDA project directory:
    python -u cluster_test_posteriors_erda.py \
        --project-root /path/to/erda/mount/master_thesis

Useful options:
    --models crf3_geneavg crf4_geneavg
    --no-resume
    --progress-every 100

Outputs are written inside the mounted project under:
    results/alternative_transcript_benchmark/test_benchmark/
    crf_retained_clusters/
"""

from __future__ import annotations

import argparse
import ast
import gc
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


MODEL_ORDER = [
    "crf3_geneavg",
    "crf3_transcriptavg",
    "crf4_geneavg",
    "crf4_transcriptavg",
]

MODEL_DISPLAY_NAMES = {
    "crf3_geneavg": "CRF 3-label, gene-average",
    "crf3_transcriptavg": "CRF 3-label, transcript-average",
    "crf4_geneavg": "CRF 4-label, gene-average",
    "crf4_transcriptavg": "CRF 4-label, transcript-average",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply frozen validation-selected clustering thresholds
            to CRF test posterior samples."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help=(
            "Mounted path to the master_thesis project root"
        ),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=MODEL_ORDER,
        help="Subset of models to process",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute models even when saved outputs already exist.",
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress every N genes.",
    )

    return parser.parse_args()


def locate_project_root(explicit_root: Path | None) -> Path:
    if explicit_root is None:
        raise ValueError("--project-root is required.")

    root = explicit_root.expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(
            f"Project root does not exist: {root}"
        )

    required_data_dir = (
        root / "data/processed/crf_paths/2026_07_08"
    )

    if not required_data_dir.is_dir():
        raise FileNotFoundError(
            "The supplied project root does not contain "
            f"{required_data_dir.relative_to(root)}: {root}"
        )

    return root


def parse_index_path(value: object) -> Tuple[int, ...]:
    if value is None:
        return tuple()

    if isinstance(value, float) and np.isnan(value):
        return tuple()

    if isinstance(value, (tuple, list, np.ndarray)):
        return tuple(int(item) for item in value)

    text = str(value).strip()

    if text.lower() in {"", "nan", "none", "[]", "()", "<na>"}:
        return tuple()

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (tuple, list)):
            return tuple(int(item) for item in parsed)

        if isinstance(parsed, (int, np.integer)):
            return (int(parsed),)

    except (ValueError, SyntaxError):
        pass

    return tuple(int(number) for number in re.findall(r"-?\d+", text))


def selected_site_f1(
    path_a: Sequence[int],
    path_b: Sequence[int],
) -> float:
    set_a = set(path_a)
    set_b = set(path_b)

    denominator = len(set_a) + len(set_b)

    if denominator == 0:
        return 1.0

    return 2.0 * len(set_a & set_b) / denominator


def cluster_gene_path_counts(
    path_count_frame: pd.DataFrame,
    n_samples: int,
    similarity_threshold: float,
) -> List[dict]:
    exact_paths = (
        path_count_frame.copy()
        .assign(path_text=lambda frame: frame["path_tuple"].map(str))
        .sort_values(
            by=["sample_count", "path_text"],
            ascending=[False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    clusters: List[dict] = []

    for row in exact_paths.itertuples(index=False):
        path = tuple(row.path_tuple)
        sample_count = int(row.sample_count)
        matching_cluster_index = None

        for cluster_index, cluster in enumerate(clusters):
            similarity = selected_site_f1(
                path,
                cluster["representative_path"],
            )

            if similarity >= similarity_threshold:
                matching_cluster_index = cluster_index
                break

        if matching_cluster_index is None:
            clusters.append(
                {
                    "representative_path": path,
                    "cluster_sample_count": sample_count,
                }
            )
        else:
            clusters[matching_cluster_index][
                "cluster_sample_count"
            ] += sample_count

    return [
        {
            "cluster_id": cluster_id,
            "representative_path": cluster["representative_path"],
            "cluster_sample_count": int(cluster["cluster_sample_count"]),
            "n_samples": int(n_samples),
            "cluster_support": (
                float(cluster["cluster_sample_count"]) / n_samples
                if n_samples > 0
                else 0.0
            ),
        }
        for cluster_id, cluster in enumerate(clusters, start=1)
    ]


def load_thresholds(
    threshold_file: Path,
    model_files: Dict[str, Path],
) -> Dict[str, dict]:
    thresholds = pd.read_csv(threshold_file, sep="\t")

    required = {
        "model_key",
        "similarity_threshold",
        "support_threshold",
    }

    missing = required - set(thresholds.columns)

    thresholds["model_key"] = thresholds["model_key"].astype(str)

    duplicate_models = thresholds["model_key"].duplicated(keep=False)
    if duplicate_models.any():
        duplicates = sorted(
            thresholds.loc[duplicate_models, "model_key"].unique()
        )

    threshold_map = {
        row.model_key: {
            "similarity_threshold": float(row.similarity_threshold),
            "support_threshold": float(row.support_threshold),
        }
        for row in thresholds.itertuples(index=False)
        if row.model_key in model_files
    }

    missing_models = set(model_files) - set(threshold_map)

    return threshold_map


def build_test_clusters_for_model(
    *,
    model_key: str,
    posterior_file: Path,
    thresholds: Dict[str, dict],
    display_names: Dict[str, str],
    output_dir: Path,
    resume: bool,
    progress_every: int,
) -> pd.DataFrame:
    similarity_threshold = float(
        thresholds[model_key]["similarity_threshold"]
    )
    support_threshold = float(
        thresholds[model_key]["support_threshold"]
    )

    retained_output_file = (
        output_dir
        / f"{model_key}_test_retained_candidate_paths.tsv"
    )

    if resume and retained_output_file.is_file():
        print(
            f"\n[{model_key}] Loading saved output: "
            f"{retained_output_file}",
            flush=True,
        )

        return pd.read_csv(
            retained_output_file,
            sep="\t",
            converters={"representative_path": parse_index_path},
        )

    print("\n" + "=" * 78, flush=True)
    print(display_names[model_key], flush=True)
    print("=" * 78, flush=True)
    print(f"Posterior file: {posterior_file}", flush=True)
    print(
        f"Similarity threshold: {similarity_threshold}",
        flush=True,
    )
    print(
        f"Support threshold: {support_threshold}",
        flush=True,
    )

    model_start = time.monotonic()

    posterior_samples = pd.read_csv(
        posterior_file,
        sep="\t",
        usecols=[
            "gene_id",
            "sample_id",
            "selected_candidate_indices",
        ],
        dtype={
            "gene_id": "string",
            "sample_id": "string",
        },
    )

    posterior_samples["gene_id"] = (
        posterior_samples["gene_id"]
        .astype(str)
        .str.strip()
    )

    duplicate_sample_rows = int(
        posterior_samples.duplicated(
            subset=["gene_id", "sample_id"]
        ).sum()
    )

    if duplicate_sample_rows > 0:
        print(
            f"Removing duplicate gene/sample rows: "
            f"{duplicate_sample_rows:,}",
            flush=True,
        )

        posterior_samples = (
            posterior_samples.drop_duplicates(
                subset=["gene_id", "sample_id"],
                keep="first",
            )
            .reset_index(drop=True)
        )

    posterior_samples["path_tuple"] = (
        posterior_samples["selected_candidate_indices"]
        .map(parse_index_path)
    )

    n_samples_by_gene = (
        posterior_samples.groupby("gene_id")["sample_id"]
        .nunique()
        .to_dict()
    )

    print(
        f"Posterior rows: {len(posterior_samples):,}",
        flush=True,
    )
    print(
        f"Posterior genes: {len(n_samples_by_gene):,}",
        flush=True,
    )

    if n_samples_by_gene:
        print(
            "Samples per gene: "
            f"{min(n_samples_by_gene.values())} to "
            f"{max(n_samples_by_gene.values())}",
            flush=True,
        )

    path_counts = (
        posterior_samples.groupby(
            ["gene_id", "path_tuple"],
            sort=False,
            as_index=False,
            dropna=False,
        )
        .size()
        .rename(columns={"size": "sample_count"})
    )

    del posterior_samples
    gc.collect()

    retained_rows: List[dict] = []
    n_genes = int(path_counts["gene_id"].nunique())

    for gene_number, (gene_id, gene_path_counts) in enumerate(
        path_counts.groupby("gene_id", sort=False),
        start=1,
    ):
        gene_clusters = cluster_gene_path_counts(
            path_count_frame=gene_path_counts[
                ["path_tuple", "sample_count"]
            ],
            n_samples=int(n_samples_by_gene[gene_id]),
            similarity_threshold=similarity_threshold,
        )

        for cluster in gene_clusters:
            representative_path = tuple(
                cluster["representative_path"]
            )
            cluster_support = float(cluster["cluster_support"])

            if cluster_support < support_threshold:
                continue

            if len(representative_path) == 0:
                continue

            retained_rows.append(
                {
                    "model_key": model_key,
                    "model": display_names[model_key],
                    "gene_id": gene_id,
                    "cluster_id": int(cluster["cluster_id"]),
                    "representative_path": representative_path,
                    "cluster_sample_count": int(
                        cluster["cluster_sample_count"]
                    ),
                    "n_samples": int(cluster["n_samples"]),
                    "cluster_support": cluster_support,
                    "similarity_threshold": similarity_threshold,
                    "support_threshold": support_threshold,
                }
            )

        if (
            progress_every > 0
            and gene_number % progress_every == 0
        ):
            elapsed_minutes = (
                time.monotonic() - model_start
            ) / 60.0

            print(
                f"[{model_key}] Clustered "
                f"{gene_number:,}/{n_genes:,} genes "
                f"({elapsed_minutes:.1f} min)",
                flush=True,
            )

    retained_columns = [
        "model_key",
        "model",
        "gene_id",
        "cluster_id",
        "representative_path",
        "cluster_sample_count",
        "n_samples",
        "cluster_support",
        "similarity_threshold",
        "support_threshold",
    ]

    retained = pd.DataFrame(
        retained_rows,
        columns=retained_columns,
    )

    if not retained.empty:
        retained = (
            retained.drop_duplicates(
                subset=["gene_id", "representative_path"]
            )
            .reset_index(drop=True)
        )

    retained.to_csv(
        retained_output_file,
        sep="\t",
        index=False,
    )

    elapsed_minutes = (
        time.monotonic() - model_start
    ) / 60.0

    print(
        f"[{model_key}] Retained paths: {len(retained):,}",
        flush=True,
    )
    print(
        f"[{model_key}] Genes with prediction: "
        f"{retained['gene_id'].nunique() if not retained.empty else 0:,}",
        flush=True,
    )
    print(
        f"[{model_key}] Finished in {elapsed_minutes:.1f} min",
        flush=True,
    )
    print(
        f"[{model_key}] Saved: {retained_output_file}",
        flush=True,
    )

    del path_counts
    del n_samples_by_gene
    gc.collect()

    return retained


def main() -> int:
    args = parse_args()
    project_root = locate_project_root(args.project_root)

    crf_root = (
        project_root
        / "data/processed/crf_paths/2026_07_08"
    )
    full_test_dir = crf_root / "full_test"

    test_model_files = {
        "crf3_geneavg": (
            full_test_dir
            / "posterior_sampling_3label_geneavg_n1000_seed13"
            / "posterior_sampled_paths.tsv"
        ),
        "crf3_transcriptavg": (
            full_test_dir
            / "posterior_sampling_3label_transcriptavg_n1000_seed13"
            / "posterior_sampled_paths.tsv"
        ),
        "crf4_geneavg": (
            full_test_dir
            / "posterior_sampling_4label_geneavg_n1000_seed13"
            / "posterior_sampled_paths.tsv"
        ),
        "crf4_transcriptavg": (
            full_test_dir
            / "posterior_sampling_4label_transcriptavg_n1000_seed13"
            / "posterior_sampled_paths.tsv"
        ),
    }

    selected_model_files = {
        model_key: test_model_files[model_key]
        for model_key in args.models
    }

    threshold_file = (
        project_root
        / "results/alternative_transcript_benchmark"
        / "validation_threshold_selection"
        / "selected_validation_cluster_thresholds.tsv"
    )

    output_dir = (
        project_root
        / "results/alternative_transcript_benchmark"
        / "test_benchmark"
        / "crf_retained_clusters"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    required_files = {
        "threshold file": threshold_file,
        **{
            f"{model_key} posterior file": path
            for model_key, path in selected_model_files.items()
        },
    }

    missing_files = [
        f"{name}: {path}"
        for name, path in required_files.items()
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(missing_files)
        )

    thresholds = load_thresholds(
        threshold_file,
        selected_model_files,
    )

    print(f"Project root: {project_root}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(
        f"Resume completed models: {not args.no_resume}",
        flush=True,
    )
    print(f"Models: {', '.join(args.models)}", flush=True)

    retained_tables: Dict[str, pd.DataFrame] = {}

    for model_key in args.models:
        retained_tables[model_key] = (
            build_test_clusters_for_model(
                model_key=model_key,
                posterior_file=selected_model_files[model_key],
                thresholds=thresholds,
                display_names=MODEL_DISPLAY_NAMES,
                output_dir=output_dir,
                resume=not args.no_resume,
                progress_every=args.progress_every,
            )
        )

    combined = pd.concat(
        list(retained_tables.values()),
        ignore_index=True,
    )

    combined_file = (
        output_dir
        / "all_crf_test_retained_candidate_paths.tsv"
    )
    combined.to_csv(combined_file, sep="\t", index=False)

    if combined.empty:
        summary = pd.DataFrame(
            columns=[
                "model_key",
                "model",
                "similarity_threshold",
                "support_threshold",
                "n_retained_paths",
                "n_genes_with_prediction",
                "mean_cluster_support",
                "median_cluster_support",
            ]
        )
    else:
        summary = (
            combined.groupby(
                [
                    "model_key",
                    "model",
                    "similarity_threshold",
                    "support_threshold",
                ],
                as_index=False,
            )
            .agg(
                n_retained_paths=(
                    "representative_path",
                    "size",
                ),
                n_genes_with_prediction=(
                    "gene_id",
                    "nunique",
                ),
                mean_cluster_support=(
                    "cluster_support",
                    "mean",
                ),
                median_cluster_support=(
                    "cluster_support",
                    "median",
                ),
            )
        )

    summary_file = output_dir / "test_cluster_summary.tsv"
    summary.to_csv(summary_file, sep="\t", index=False)

    print("\nSummary:", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nSaved combined output: {combined_file}", flush=True)
    print(f"Saved summary: {summary_file}", flush=True)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"\nERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise
