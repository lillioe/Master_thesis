#!/usr/bin/env python

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torchcrf import CRF


NEG_INF = -10000.0


class SplicePathCRFInference(nn.Module):
    def __init__(self, label_mode=4, enforce_transition_constraints=True):
        super().__init__()
        self.label_mode = int(label_mode)
        self.num_tags = 4 if self.label_mode == 4 else 3
        self.enforce_transition_constraints = bool(enforce_transition_constraints)

        self.logit_scale = nn.Parameter(torch.ones(2))
        self.logit_bias = nn.Parameter(torch.zeros(2))
        self.crf = CRF(self.num_tags, batch_first=True)

        if self.label_mode == 4:
            mask = torch.full((4, 4), NEG_INF)
            # 0=S_D, 1=D, 2=S_A, 3=A
            valid = [
                (0, 0), (0, 1),
                (1, 2), (1, 3),
                (2, 2), (2, 3),
                (3, 0), (3, 1),
            ]
            for i, j in valid:
                mask[i, j] = 0.0
        else:
            mask = torch.zeros((3, 3))

        self.register_buffer("transition_mask", mask)

    def calibrated_logits(self, logit_not_used, logit_used):
        x = torch.stack([logit_not_used, logit_used], dim=-1)
        return x * self.logit_scale + self.logit_bias

    def make_emissions(self, logit_not_used, logit_used, candidate_type):
        x = self.calibrated_logits(logit_not_used, logit_used)
        not_used = x[:, 0]
        used = x[:, 1]

        n = len(candidate_type)

        if self.label_mode == 4:
            # labels: 0=S_D, 1=D, 2=S_A, 3=A
            emissions = torch.full((n, 4), NEG_INF, dtype=x.dtype, device=x.device)

            is_donor = torch.tensor(
                [str(t).lower() == "donor" for t in candidate_type],
                dtype=torch.bool,
                device=x.device,
            )
            is_acceptor = torch.tensor(
                [str(t).lower() == "acceptor" for t in candidate_type],
                dtype=torch.bool,
                device=x.device,
            )

            emissions[:, 0] = not_used
            emissions[:, 2] = not_used
            emissions[is_donor, 1] = used[is_donor]
            emissions[is_acceptor, 3] = used[is_acceptor]

        else:
            # labels: 0=skip, 1=donor, 2=acceptor
            emissions = torch.full((n, 3), NEG_INF, dtype=x.dtype, device=x.device)

            is_donor = torch.tensor(
                [str(t).lower() == "donor" for t in candidate_type],
                dtype=torch.bool,
                device=x.device,
            )
            is_acceptor = torch.tensor(
                [str(t).lower() == "acceptor" for t in candidate_type],
                dtype=torch.bool,
                device=x.device,
            )

            emissions[:, 0] = not_used
            emissions[is_donor, 1] = used[is_donor]
            emissions[is_acceptor, 2] = used[is_acceptor]

        return emissions

    def decode_one_gene(self, logit_not_used, logit_used, candidate_type):
        emissions = self.make_emissions(logit_not_used, logit_used, candidate_type)

        if self.enforce_transition_constraints:
            old_transitions = self.crf.transitions.data.clone()
            self.crf.transitions.data = self.crf.transitions.data + self.transition_mask

        path = self.crf.decode(emissions.unsqueeze(0))[0]

        if self.enforce_transition_constraints:
            self.crf.transitions.data = old_transitions

        return path


def parse_selected_indices(x):
    if pd.isna(x):
        return set()
    s = str(x)
    if s.strip() == "":
        return set()
    out = set()
    for part in s.replace(",", ";").split(";"):
        part = part.strip()
        if part == "":
            continue
        try:
            out.add(int(part))
        except ValueError:
            pass
    return out


def selected_from_labels(labels, candidate_indices):
    selected = set()
    for lab, idx in zip(labels, candidate_indices):
        if lab in (1, 3):
            selected.add(int(idx))
    return selected


def label_to_selected_type(label, label_mode):
    if label_mode == 4:
        if label == 1:
            return "donor"
        if label == 3:
            return "acceptor"
        return "skip"
    else:
        if label == 1:
            return "donor"
        if label == 2:
            return "acceptor"
        return "skip"


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nt-logits", required=True)
    ap.add_argument("--transcript-paths", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split-name", default="")
    ap.add_argument("--copy-checkpoint", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    label_mode = int(ckpt.get("label_mode", 4))
    enforce = bool(ckpt.get("enforce_transition_constraints", label_mode == 4))

    print("label_mode:", label_mode)
    print("enforce_transition_constraints:", enforce)

    model = SplicePathCRFInference(
        label_mode=label_mode,
        enforce_transition_constraints=enforce,
    )
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()

    print("Loading transcript paths")
    paths = pd.read_csv(args.transcript_paths, sep="\t")
    paths = paths[paths["path_valid"].astype(int) == 1].copy()

    gold_by_gene = {}
    for gene_id, g in paths.groupby("gene_id", sort=False):
        gold_by_gene[str(gene_id)] = [
            parse_selected_indices(x) for x in g["selected_candidate_indices"]
        ]


    pred_rows = []
    cand_rows = []
    gene_metric_rows = []

    total_genes = 0
    total_candidates = 0

    tp_total = fp_total = fn_total = 0
    exact_any_total = 0
    genes_with_gold = 0

    usecols = None
    df_iter = pd.read_csv(args.nt_logits, sep="\t", chunksize=1_000_000, low_memory=False)

    carry = pd.DataFrame()

    def process_gene(gene_df):
        nonlocal total_genes, total_candidates
        nonlocal tp_total, fp_total, fn_total, exact_any_total, genes_with_gold

        gene_df = gene_df.sort_values("candidate_index", kind="mergesort").copy()
        gene_id = str(gene_df["gene_id"].iloc[0])

        candidate_indices = gene_df["candidate_index"].astype(int).tolist()
        candidate_types = gene_df["candidate_type"].astype(str).tolist()

        logit_not_used = torch.tensor(
            gene_df["logit_not_used"].astype(float).values,
            dtype=torch.float32,
            device=device,
        )
        logit_used = torch.tensor(
            gene_df["logit_used"].astype(float).values,
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            labels = model.decode_one_gene(logit_not_used, logit_used, candidate_types)

        selected = selected_from_labels(labels, candidate_indices)
        gold_paths = gold_by_gene.get(gene_id, [])

        best_f1 = 0.0
        best_jaccard = 0.0
        exact_any = 0
        best_gold_size = 0

        if gold_paths:
            genes_with_gold += 1
            for gold in gold_paths:
                tp = len(selected & gold)
                fp = len(selected - gold)
                fn = len(gold - selected)
                _, _, f1 = prf(tp, fp, fn)
                union = len(selected | gold)
                jac = len(selected & gold) / union if union else 1.0

                if f1 > best_f1:
                    best_f1 = f1
                    best_gold_size = len(gold)
                if jac > best_jaccard:
                    best_jaccard = jac
                if selected == gold:
                    exact_any = 1

            # compare to the best match annotated path
            best_gold = max(
                gold_paths,
                key=lambda gold: len(selected & gold) / len(selected | gold)
                if len(selected | gold)
                else 1.0,
            )
            tp = len(selected & best_gold)
            fp = len(selected - best_gold)
            fn = len(best_gold - selected)
            tp_total += tp
            fp_total += fp
            fn_total += fn
            exact_any_total += exact_any

        pred_rows.append({
            "gene_id": gene_id,
            "n_candidates": len(gene_df),
            "n_selected": len(selected),
            "selected_candidate_indices": ";".join(map(str, sorted(selected))),
            "has_gold": int(bool(gold_paths)),
            "n_gold_paths": len(gold_paths),
            "exact_path_match_any": exact_any,
            "best_selected_f1": best_f1,
            "best_selected_jaccard": best_jaccard,
            "best_gold_size": best_gold_size,
        })

        gene_metric_rows.append(pred_rows[-1].copy())

        for row, lab in zip(gene_df.itertuples(index=False), labels):
            selected_type = label_to_selected_type(int(lab), label_mode)
            cand_rows.append({
                "gene_id": gene_id,
                "seqid": getattr(row, "seqid", None),
                "strand": getattr(row, "strand", None),
                "candidate_index": int(getattr(row, "candidate_index")),
                "candidate_type": getattr(row, "candidate_type"),
                "relative_base_pos": getattr(row, "relative_base_pos", None),
                "genomic_pos_1based": getattr(row, "genomic_pos_1based", None),
                "crf_label": int(lab),
                "crf_selected_type": selected_type,
                "crf_selected": int(selected_type != "skip"),
            })

        total_genes += 1
        total_candidates += len(gene_df)

    for chunk in df_iter:
        if len(carry):
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = pd.DataFrame()

        # Assumes nt_logits is sorted/grouped by gene_id, as produced by export script.
        gene_ids = chunk["gene_id"].astype(str)
        last_gene = gene_ids.iloc[-1]

        complete = chunk[gene_ids != last_gene]
        carry = chunk[gene_ids == last_gene].copy()

        for _, g in complete.groupby("gene_id", sort=False):
            process_gene(g)

    if len(carry):
        for _, g in carry.groupby("gene_id", sort=False):
            process_gene(g)

    pred_df = pd.DataFrame(pred_rows)
    cand_df = pd.DataFrame(cand_rows)
    metric_df = pd.DataFrame(gene_metric_rows)

    pred_df.to_csv(out_dir / "viterbi_gene_predictions.tsv", sep="\t", index=False)
    cand_df.to_csv(out_dir / "viterbi_candidate_predictions.tsv", sep="\t", index=False)
    metric_df.to_csv(out_dir / "metrics_by_gene.tsv", sep="\t", index=False)

    precision, recall, f1 = prf(tp_total, fp_total, fn_total)

    summary = {
        "split_name": args.split_name,
        "checkpoint": args.checkpoint,
        "nt_logits": args.nt_logits,
        "transcript_paths": args.transcript_paths,
        "label_mode": label_mode,
        "enforce_transition_constraints": enforce,
        "n_genes_decoded": total_genes,
        "n_candidates_decoded": total_candidates,
        "n_genes_with_gold": genes_with_gold,
        "exact_path_match_any_mean": exact_any_total / genes_with_gold if genes_with_gold else None,
        "mean_best_selected_f1": float(metric_df["best_selected_f1"].mean()) if len(metric_df) else None,
        "mean_best_selected_jaccard": float(metric_df["best_selected_jaccard"].mean()) if len(metric_df) else None,
        "selected_tp_total_best_gold": tp_total,
        "selected_fp_total_best_gold": fp_total,
        "selected_fn_total_best_gold": fn_total,
        "selected_precision_best_gold": precision,
        "selected_recall_best_gold": recall,
        "selected_f1_best_gold": f1,
    }

    pd.DataFrame([summary]).to_csv(out_dir / "metrics_summary.tsv", sep="\t", index=False)

    with open(out_dir / "config.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.copy_checkpoint:
        shutil.copy2(args.checkpoint, out_dir / "checkpoint_used.pt")

    print(json.dumps(summary, indent=2))
    print("Wrote:", out_dir)


if __name__ == "__main__":
    main()
