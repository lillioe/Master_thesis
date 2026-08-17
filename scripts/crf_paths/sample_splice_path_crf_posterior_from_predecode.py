#!/usr/bin/env python

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def logsumexp(a, axis=None):
    a = np.asarray(a, dtype=np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def softmax_sample(log_probs, rng):
    log_probs = np.asarray(log_probs, dtype=np.float64)
    m = np.max(log_probs)
    probs = np.exp(log_probs - m)
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))


def parse_list(x):
    if pd.isna(x) or str(x).strip() == "":
        return []
    return [int(v.strip()) for v in str(x).replace(",", ";").split(";") if v.strip()]


def parse_types(x):
    if pd.isna(x) or str(x).strip() == "":
        return []
    return [v.strip() for v in str(x).replace(",", ";").split(";") if v.strip()]


def make_junctions(indices, types):
    out = []
    for i in range(len(indices) - 1):
        if types[i] == "donor" and types[i + 1] == "acceptor":
            out.append((indices[i], indices[i + 1]))
    return set(out)


def prf(pred_set, gold_set):
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    union = len(pred_set | gold_set)
    jaccard = tp / union if union else 1.0

    return precision, recall, f1, jaccard


def selected_from_labels(labels, candidate_indices, candidate_types, label_mode):
    selected_indices = []
    selected_types = []

    for lab, idx, typ in zip(labels, candidate_indices, candidate_types):
        if label_mode == 4:
            if lab == 1 or lab == 3:
                selected_indices.append(int(idx))
                selected_types.append(str(typ))
        else:
            if lab == 1 or lab == 2:
                selected_indices.append(int(idx))
                selected_types.append(str(typ))

    return selected_indices, selected_types


def load_gold(transcript_paths):
    gold = pd.read_csv(transcript_paths, sep="\t")
    if "path_valid" in gold.columns:
        gold = gold[gold["path_valid"].astype(int).eq(1)].copy()

    gold_by_gene = {}

    for r in gold.itertuples(index=False):
        indices = parse_list(r.selected_candidate_indices)
        types = parse_types(r.selected_candidate_types)
        gene_id = str(r.gene_id)

        gold_by_gene.setdefault(gene_id, []).append({
            "transcript_id": getattr(r, "transcript_id", ""),
            "indices": indices,
            "types": types,
            "set": set(indices),
            "junctions": make_junctions(indices, types),
        })

    return gold_by_gene


def forward_algorithm(emissions, transitions, start, end):
    n, n_tags = emissions.shape
    alpha = np.full((n, n_tags), -np.inf, dtype=np.float64)

    alpha[0] = start + emissions[0]

    for i in range(1, n):
        for curr in range(n_tags):
            alpha[i, curr] = emissions[i, curr] + logsumexp(alpha[i - 1] + transitions[:, curr])

    log_z = logsumexp(alpha[-1] + end)
    return alpha, log_z


def sample_path(alpha, emissions, transitions, end, rng):
    n, n_tags = alpha.shape
    labels = np.empty(n, dtype=np.int16)

    labels[-1] = softmax_sample(alpha[-1] + end, rng)

    for i in range(n - 1, 0, -1):
        curr = labels[i]
        # emission at current state is constant with respect to previous tag.
        prev_scores = alpha[i - 1] + transitions[:, curr]
        labels[i - 1] = softmax_sample(prev_scores, rng)

    return labels


def evaluate_sample(selected_indices, selected_types, gold_paths):
    pred_set = set(selected_indices)
    pred_junc = make_junctions(selected_indices, selected_types)

    exact_any = 0
    best_site_f1 = 0.0
    best_site_jaccard = 0.0
    best_junction_f1 = 0.0
    best_junction_jaccard = 0.0

    for gold in gold_paths:
        if selected_indices == gold["indices"]:
            exact_any = 1

        _, _, site_f1, site_j = prf(pred_set, gold["set"])
        _, _, junc_f1, junc_j = prf(pred_junc, gold["junctions"])

        best_site_f1 = max(best_site_f1, site_f1)
        best_site_jaccard = max(best_site_jaccard, site_j)
        best_junction_f1 = max(best_junction_f1, junc_f1)
        best_junction_jaccard = max(best_junction_jaccard, junc_j)

    return exact_any, best_site_f1, best_site_jaccard, best_junction_f1, best_junction_jaccard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-emissions", required=True)
    ap.add_argument("--model-params", required=True)
    ap.add_argument("--transcript-paths", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    params = torch.load(args.model_params, map_location="cpu", weights_only=False)
    label_mode = int(params["label_mode"])

    start = params["start_transitions"].detach().cpu().numpy().astype("float64")
    end = params["end_transitions"].detach().cpu().numpy().astype("float64")
    transitions = params["transitions_effective"].detach().cpu().numpy().astype("float64")

    if label_mode == 4:
        emission_cols = ["emit_S_D", "emit_D", "emit_S_A", "emit_A"]
    else:
        emission_cols = ["emit_skip", "emit_D", "emit_A"]

    gold_by_gene = load_gold(args.transcript_paths)

    df = pd.read_csv(args.candidate_emissions, sep="\t", low_memory=False)
    print("rows:", len(df))
    print("genes:", df["gene_id"].nunique())
    print("samples per gene:", args.n_samples)

    sample_rows = []
    gene_metric_rows = []
    candidate_freq_rows = []

    for gi, (gene_id, g) in enumerate(df.groupby("gene_id", sort=False), start=1):
        g = g.sort_values("candidate_index", kind="mergesort").copy()

        candidate_indices = g["candidate_index"].astype(int).tolist()
        candidate_types = g["candidate_type"].astype(str).tolist()
        emissions = g[emission_cols].astype("float64").to_numpy()

        alpha, log_z = forward_algorithm(emissions, transitions, start, end)

        gold_paths = gold_by_gene.get(str(gene_id), [])

        exact_any_over_samples = 0
        best_site_f1_over_samples = 0.0
        best_site_jaccard_over_samples = 0.0
        best_junction_f1_over_samples = 0.0
        best_junction_jaccard_over_samples = 0.0

        unique_paths = set()
        selected_counter = Counter()

        for sample_id in range(1, args.n_samples + 1):
            labels = sample_path(alpha, emissions, transitions, end, rng)
            selected_indices, selected_types = selected_from_labels(
                labels,
                candidate_indices,
                candidate_types,
                label_mode,
            )

            key = ";".join(map(str, selected_indices))
            unique_paths.add(key)

            for idx, typ in zip(selected_indices, selected_types):
                selected_counter[(idx, typ)] += 1

            exact_any = 0
            site_f1 = site_jaccard = junction_f1 = junction_jaccard = 0.0

            if gold_paths:
                exact_any, site_f1, site_jaccard, junction_f1, junction_jaccard = evaluate_sample(
                    selected_indices,
                    selected_types,
                    gold_paths,
                )

                exact_any_over_samples = max(exact_any_over_samples, exact_any)
                best_site_f1_over_samples = max(best_site_f1_over_samples, site_f1)
                best_site_jaccard_over_samples = max(best_site_jaccard_over_samples, site_jaccard)
                best_junction_f1_over_samples = max(best_junction_f1_over_samples, junction_f1)
                best_junction_jaccard_over_samples = max(best_junction_jaccard_over_samples, junction_jaccard)

            sample_rows.append({
                "gene_id": gene_id,
                "sample_id": sample_id,
                "log_z": float(log_z),
                "n_selected": len(selected_indices),
                "selected_candidate_indices": ";".join(map(str, selected_indices)),
                "selected_candidate_types": ";".join(map(str, selected_types)),
                "exact_path_match_any": exact_any,
                "best_selected_f1": site_f1,
                "best_selected_jaccard": site_jaccard,
                "best_junction_f1": junction_f1,
                "best_junction_jaccard": junction_jaccard,
            })

        for (idx, typ), count in selected_counter.items():
            candidate_freq_rows.append({
                "gene_id": gene_id,
                "candidate_index": idx,
                "candidate_type": typ,
                "sample_count_selected": count,
                "sample_frequency_selected": count / args.n_samples,
            })

        gene_metric_rows.append({
            "gene_id": gene_id,
            "n_samples": args.n_samples,
            "n_unique_sampled_paths": len(unique_paths),
            "sample_exact_path_match_any": exact_any_over_samples,
            "best_selected_f1_across_samples": best_site_f1_over_samples,
            "best_selected_jaccard_across_samples": best_site_jaccard_over_samples,
            "best_junction_f1_across_samples": best_junction_f1_over_samples,
            "best_junction_jaccard_across_samples": best_junction_jaccard_over_samples,
            "has_gold": int(bool(gold_paths)),
            "n_gold_paths": len(gold_paths),
        })

        if gi % 10 == 0:
            print(f"sampled genes={gi}")

    samples = pd.DataFrame(sample_rows)
    gene_metrics = pd.DataFrame(gene_metric_rows)
    candidate_freq = pd.DataFrame(candidate_freq_rows)

    samples.to_csv(out_dir / "posterior_sampled_paths.tsv", sep="\t", index=False)
    gene_metrics.to_csv(out_dir / "posterior_metrics_by_gene.tsv", sep="\t", index=False)
    candidate_freq.to_csv(out_dir / "candidate_sample_frequencies.tsv", sep="\t", index=False)

    gold_gene_metrics = gene_metrics[gene_metrics["has_gold"].astype(int).eq(1)].copy()

    summary = pd.DataFrame([{
        "n_genes": len(gene_metrics),
        "n_genes_with_gold": len(gold_gene_metrics),
        "n_samples": args.n_samples,
        "mean_unique_sampled_paths": gene_metrics["n_unique_sampled_paths"].mean(),
        "sample_exact_path_match_any_mean": gold_gene_metrics["sample_exact_path_match_any"].mean(),
        "mean_best_selected_f1_across_samples": gold_gene_metrics["best_selected_f1_across_samples"].mean(),
        "mean_best_selected_jaccard_across_samples": gold_gene_metrics["best_selected_jaccard_across_samples"].mean(),
        "mean_best_junction_f1_across_samples": gold_gene_metrics["best_junction_f1_across_samples"].mean(),
        "mean_best_junction_jaccard_across_samples": gold_gene_metrics["best_junction_jaccard_across_samples"].mean(),
    }])

    summary.to_csv(out_dir / "posterior_metrics_summary.tsv", sep="\t", index=False)

    print(summary.to_string(index=False))
    print("Wrote:", out_dir)


if __name__ == "__main__":
    main()
