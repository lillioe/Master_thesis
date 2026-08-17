#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


NEG_INF = -10000.0


def get_state_dict(ckpt):
    return ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))


def make_transition_mask(label_mode):
    if label_mode == 4:
        mask = torch.full((4, 4), NEG_INF)
        valid = [
            (0, 0), (0, 1),  # S_D -> S_D or D
            (1, 2), (1, 3),  # D   -> S_A or A
            (2, 2), (2, 3),  # S_A -> S_A or A
            (3, 0), (3, 1),  # A   -> S_D or D
        ]
        for i, j in valid:
            mask[i, j] = 0.0
        return mask

    if label_mode == 3:
        return torch.zeros((3, 3))


def add_emission_columns(chunk, label_mode, logit_scale, logit_bias):
    logit_not = chunk["logit_not_used"].astype("float32")
    logit_used = chunk["logit_used"].astype("float32")

    not_used = logit_not * float(logit_scale[0]) + float(logit_bias[0])
    used = logit_used * float(logit_scale[1]) + float(logit_bias[1])

    cand_type = chunk["candidate_type"].astype(str).str.lower()
    is_donor = cand_type.eq("donor")
    is_acceptor = cand_type.eq("acceptor")

    if label_mode == 4:
        # 4-label CRF:
        # 0 = S_D, skipped while waiting for donor
        # 1 = D, selected donor
        # 2 = S_A, skipped while waiting for acceptor
        # 3 = A, selected acceptor
        chunk["emit_S_D"] = not_used
        chunk["emit_D"] = NEG_INF
        chunk.loc[is_donor, "emit_D"] = used[is_donor]

        chunk["emit_S_A"] = not_used
        chunk["emit_A"] = NEG_INF
        chunk.loc[is_acceptor, "emit_A"] = used[is_acceptor]

    else:
        # 3-label CRF:
        # 0 = skip
        # 1 = donor
        # 2 = acceptor
        chunk["emit_skip"] = not_used

        chunk["emit_D"] = NEG_INF
        chunk.loc[is_donor, "emit_D"] = used[is_donor]

        chunk["emit_A"] = NEG_INF
        chunk.loc[is_acceptor, "emit_A"] = used[is_acceptor]

    return chunk


def update_gene_offsets(gene_ids, gene_offsets, state):
    # Assumes nt_logits.tsv is grouped by gene_id
    for gene_id in gene_ids:
        gene_id = str(gene_id)

        if state["current_gene"] is None:
            state["current_gene"] = gene_id
            state["current_start"] = state["row_index"]
            state["current_count"] = 1

        elif gene_id == state["current_gene"]:
            state["current_count"] += 1

        else:
            gene_offsets.append({
                "gene_id": state["current_gene"],
                "start_row": state["current_start"],
                "n_candidates": state["current_count"],
            })

            state["current_gene"] = gene_id
            state["current_start"] = state["row_index"]
            state["current_count"] = 1

        state["row_index"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt-logits", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--transcript-paths", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split-name", default="")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoint")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = get_state_dict(ckpt)

    label_mode = int(ckpt.get("label_mode", 4))
    enforce = bool(ckpt.get("enforce_transition_constraints", label_mode == 4))

    logit_scale = sd["logit_scale"].detach().cpu()
    logit_bias = sd["logit_bias"].detach().cpu()

    start_transitions = sd["crf.start_transitions"].detach().cpu()
    end_transitions = sd["crf.end_transitions"].detach().cpu()
    transitions_raw = sd["crf.transitions"].detach().cpu()

    transition_mask = make_transition_mask(label_mode)
    transitions_effective = transitions_raw + transition_mask if enforce else transitions_raw

    torch.save(
        {
            "label_mode": label_mode,
            "enforce_transition_constraints": enforce,
            "logit_scale": logit_scale,
            "logit_bias": logit_bias,
            "start_transitions": start_transitions,
            "end_transitions": end_transitions,
            "transitions_raw": transitions_raw,
            "transition_mask": transition_mask,
            "transitions_effective": transitions_effective,
        },
        out_dir / "crf_model_params.pt",
    )

    config = {
        "split_name": args.split_name,
        "nt_logits": args.nt_logits,
        "checkpoint": args.checkpoint,
        "transcript_paths": args.transcript_paths,
        "label_mode": label_mode,
        "enforce_transition_constraints": enforce,
        "note": "Pre-decoding CRF output. No Viterbi, top-k, marginals, or posterior decoding was performed.",
    }

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Preparing CRF candidate emissions")
    print("label_mode:", label_mode)

    emissions_path = out_dir / "crf_candidate_emissions.tsv"
    gene_offsets = []
    seen_genes = set()
    first_write = True

    state = {
        "current_gene": None,
        "current_start": 0,
        "current_count": 0,
        "row_index": 0,
    }

    for i, chunk in enumerate(
        pd.read_csv(args.nt_logits, sep="\t", chunksize=args.chunksize, low_memory=False),
        start=1,
    ):
        seen_genes.update(chunk["gene_id"].astype(str).unique())

        update_gene_offsets(
            chunk["gene_id"].astype(str).tolist(),
            gene_offsets,
            state,
        )

        chunk = add_emission_columns(
            chunk=chunk,
            label_mode=label_mode,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
        )

        chunk.to_csv(
            emissions_path,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )
        first_write = False

        print(
            f"processed chunks={i} "
            f"rows={state['row_index']} "
            f"genes_seen={len(seen_genes)}",
            flush=True,
        )

    if state["current_gene"] is not None:
        gene_offsets.append({
            "gene_id": state["current_gene"],
            "start_row": state["current_start"],
            "n_candidates": state["current_count"],
        })

    pd.DataFrame(gene_offsets).to_csv(
        out_dir / "crf_gene_offsets.tsv",
        sep="\t",
        index=False,
    )

    if args.transcript_paths:
        print("Writing gold transcript paths for genes present in logits...")
        paths = pd.read_csv(args.transcript_paths, sep="\t")
        paths = paths[paths["gene_id"].astype(str).isin(seen_genes)].copy()
        
        paths.to_csv(
            out_dir / "gold_paths_by_gene.tsv",
            sep="\t",
            index=False,
        )

    summary = {
        "split_name": args.split_name,
        "n_rows": state["row_index"],
        "n_genes": len(gene_offsets),
        "label_mode": label_mode,
        "enforce_transition_constraints": enforce,
        "candidate_emissions": str(emissions_path),
        "gene_offsets": str(out_dir / "crf_gene_offsets.tsv"),
        "model_params": str(out_dir / "crf_model_params.pt"),
        "gold_paths": str(out_dir / "gold_paths_by_gene.tsv") if args.transcript_paths else None,
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("Wrote:", out_dir)


if __name__ == "__main__":
    main()
