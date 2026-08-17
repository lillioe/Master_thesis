#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


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


def topk_viterbi_unique(emissions, transitions, start, end, candidate_indices, candidate_types, label_mode, k_max, k_internal):
    n, n_tags = emissions.shape
    neg = -1e30

    scores = np.full((n, n_tags, k_internal), neg, dtype=np.float32)
    prev_tag = np.full((n, n_tags, k_internal), -1, dtype=np.int16)
    prev_rank = np.full((n, n_tags, k_internal), -1, dtype=np.int16)

    scores[0, :, 0] = start + emissions[0]

    for i in range(1, n):
        for curr in range(n_tags):
            cand = scores[i - 1] + transitions[:, curr][:, None]
            flat = cand.reshape(-1)

            valid = np.isfinite(flat) & (flat > neg / 2)
            if not valid.any():
                continue

            valid_idx = np.where(valid)[0]
            valid_scores = flat[valid_idx]

            take = min(k_internal, len(valid_scores))
            if take < len(valid_scores):
                part = np.argpartition(valid_scores, -take)[-take:]
                order = part[np.argsort(valid_scores[part])[::-1]]
            else:
                order = np.argsort(valid_scores)[::-1]

            chosen_flat_idx = valid_idx[order]
            chosen_scores = flat[chosen_flat_idx] + emissions[i, curr]

            tags = chosen_flat_idx // k_internal
            ranks = chosen_flat_idx % k_internal

            scores[i, curr, :take] = chosen_scores[:take]
            prev_tag[i, curr, :take] = tags[:take]
            prev_rank[i, curr, :take] = ranks[:take]

    final = scores[-1] + end[:, None]
    flat = final.reshape(-1)
    valid = np.isfinite(flat) & (flat > neg / 2)

    if not valid.any():
        return []

    valid_idx = np.where(valid)[0]
    valid_scores = flat[valid_idx]

    take = min(k_internal * n_tags, len(valid_scores))
    order = np.argsort(valid_scores)[::-1][:take]
    chosen_final = valid_idx[order]

    paths = []
    seen_selected = set()

    for flat_idx in chosen_final:
        tag = int(flat_idx // k_internal)
        rank = int(flat_idx % k_internal)
        score = float(flat[flat_idx])

        labels = np.empty(n, dtype=np.int16)
        labels[-1] = tag

        cur_tag = tag
        cur_rank = rank

        ok = True
        for i in range(n - 1, 0, -1):
            pt = int(prev_tag[i, cur_tag, cur_rank])
            pr = int(prev_rank[i, cur_tag, cur_rank])
            if pt < 0 or pr < 0:
                ok = False
                break
            labels[i - 1] = pt
            cur_tag, cur_rank = pt, pr

        if not ok:
            continue

        selected_indices, selected_types = selected_from_labels(
            labels,
            candidate_indices,
            candidate_types,
            label_mode,
        )

        key = ";".join(map(str, selected_indices))
        if key in seen_selected:
            continue

        seen_selected.add(key)

        paths.append({
            "path_score": score,
            "labels": labels,
            "selected_indices": selected_indices,
            "selected_types": selected_types,
        })

        if len(paths) >= k_max:
            break

    return paths


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


def evaluate_gene(gene_id, pred_paths, gold_paths, k_values):
    rows = []

    for K in k_values:
        use_paths = pred_paths[:K]

        exact_any = 0
        first_exact_rank = np.nan
        best_selected_f1 = 0.0
        best_selected_jaccard = 0.0
        best_junction_f1 = 0.0
        best_junction_jaccard = 0.0

        for pred in use_paths:
            pred_set = set(pred["selected_indices"])
            pred_junc = make_junctions(pred["selected_indices"], pred["selected_types"])

            for gold in gold_paths:
                if pred["selected_indices"] == gold["indices"]:
                    exact_any = 1
                    if pd.isna(first_exact_rank):
                        first_exact_rank = pred["rank"]

                _, _, site_f1, site_j = prf(pred_set, gold["set"])
                _, _, junc_f1, junc_j = prf(pred_junc, gold["junctions"])

                best_selected_f1 = max(best_selected_f1, site_f1)
                best_selected_jaccard = max(best_selected_jaccard, site_j)
                best_junction_f1 = max(best_junction_f1, junc_f1)
                best_junction_jaccard = max(best_junction_jaccard, junc_j)

        rows.append({
            "gene_id": gene_id,
            "K": K,
            "exact_path_match_any_at_K": exact_any,
            "first_exact_rank": first_exact_rank,
            "best_selected_f1_at_K": best_selected_f1,
            "best_selected_jaccard_at_K": best_selected_jaccard,
            "best_junction_f1_at_K": best_junction_f1,
            "best_junction_jaccard_at_K": best_junction_jaccard,
            "n_gold_paths": len(gold_paths),
            "n_pred_paths_considered": len(use_paths),
        })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-emissions", required=True)
    ap.add_argument("--model-params", required=True)
    ap.add_argument("--transcript-paths", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k-max", type=int, default=50)
    ap.add_argument("--k-values", default="1,10,15,30,50")
    ap.add_argument("--k-internal", type=int, default=150)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k_values = [int(x) for x in args.k_values.split(",")]
    k_max = max(max(k_values), args.k_max)
    k_internal = max(k_max, args.k_internal)

    params = torch.load(args.model_params, map_location="cpu", weights_only=False)
    label_mode = int(params["label_mode"])

    start = params["start_transitions"].detach().cpu().numpy().astype("float32")
    end = params["end_transitions"].detach().cpu().numpy().astype("float32")
    transitions = params["transitions_effective"].detach().cpu().numpy().astype("float32")

    if label_mode == 4:
        emission_cols = ["emit_S_D", "emit_D", "emit_S_A", "emit_A"]
    else:
        emission_cols = ["emit_skip", "emit_D", "emit_A"]

    gold_by_gene = load_gold(args.transcript_paths)

    print("Loading candidate emissions")
    df = pd.read_csv(args.candidate_emissions, sep="\t", low_memory=False)
    print("rows:", len(df))
    print("genes:", df["gene_id"].nunique())
    print("label_mode:", label_mode)
    print("k_max:", k_max)
    print("k_internal:", k_internal)

    pred_rows = []
    metric_rows = []

    for gi, (gene_id, g) in enumerate(df.groupby("gene_id", sort=False), start=1):
        g = g.sort_values("candidate_index", kind="mergesort").copy()

        candidate_indices = g["candidate_index"].astype(int).tolist()
        candidate_types = g["candidate_type"].astype(str).tolist()
        emissions = g[emission_cols].astype("float32").to_numpy()

        paths = topk_viterbi_unique(
            emissions=emissions,
            transitions=transitions,
            start=start,
            end=end,
            candidate_indices=candidate_indices,
            candidate_types=candidate_types,
            label_mode=label_mode,
            k_max=k_max,
            k_internal=k_internal,
        )

        for rank, p in enumerate(paths, start=1):
            p["rank"] = rank

            pred_rows.append({
                "gene_id": gene_id,
                "rank": rank,
                "path_score": p["path_score"],
                "n_selected": len(p["selected_indices"]),
                "selected_candidate_indices": ";".join(map(str, p["selected_indices"])),
                "selected_candidate_types": ";".join(map(str, p["selected_types"])),
            })

        gold_paths = gold_by_gene.get(str(gene_id), [])
        if gold_paths:
            metric_rows.extend(evaluate_gene(str(gene_id), paths, gold_paths, k_values))

        if gi % 10 == 0:
            print(f"decoded genes={gi}")

    pred = pd.DataFrame(pred_rows)
    metrics = pd.DataFrame(metric_rows)

    pred.to_csv(out_dir / "topk_gene_predictions.tsv", sep="\t", index=False)
    metrics.to_csv(out_dir / "kbest_metrics_by_gene.tsv", sep="\t", index=False)

    summary = (
        metrics.groupby("K")
        .agg(
            n_genes=("gene_id", "nunique"),
            exact_path_match_any_at_K=("exact_path_match_any_at_K", "mean"),
            genes_with_exact_match=("exact_path_match_any_at_K", "sum"),
            mean_best_selected_f1_at_K=("best_selected_f1_at_K", "mean"),
            mean_best_selected_jaccard_at_K=("best_selected_jaccard_at_K", "mean"),
            mean_best_junction_f1_at_K=("best_junction_f1_at_K", "mean"),
            mean_best_junction_jaccard_at_K=("best_junction_jaccard_at_K", "mean"),
            median_first_exact_rank=("first_exact_rank", "median"),
        )
        .reset_index()
    )

    summary.to_csv(out_dir / "kbest_metrics_summary.tsv", sep="\t", index=False)

    print(summary.to_string(index=False))
    print("Wrote:", out_dir)


if __name__ == "__main__":
    main()
