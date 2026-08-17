from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.models.nt_pool_transformer import NTPoolCandidateTransformer


def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, sep="\t", nrows=0).columns.tolist()


def build_position_features(pos, candidate_types):
    pos = torch.tensor(pos, dtype=torch.float32)

    if len(pos) > 1 and float(pos.max() - pos.min()) > 0:
        pos_norm = (pos - pos.min()) / (pos.max() - pos.min())
    else:
        pos_norm = torch.zeros_like(pos)

    is_donor = torch.tensor(
        [1.0 if str(t) == "donor" else 0.0 for t in candidate_types],
        dtype=torch.float32,
    )

    is_acceptor = torch.tensor(
        [1.0 if str(t) == "acceptor" else 0.0 for t in candidate_types],
        dtype=torch.float32,
    )

    return torch.stack([pos_norm, is_donor, is_acceptor], dim=1)


def iter_gene_groups_sorted(
    path: Path,
    usecols: list[str],
    group_col: str,
    pos_col: str,
    candidate_index_col: str,
    chunksize: int,
):
    """
    Stream one complete gene at a time from TSV sorted by gene_id
    input must be sorted so all rows for a gene are contiguous
    """

    current_gene = None
    buffer = []

    reader = pd.read_csv(path, sep="\t", usecols=usecols, chunksize=chunksize)

    for chunk in reader:
        for row in chunk.itertuples(index=False):
            d = row._asdict()
            gene_id = str(d[group_col])

            if current_gene is None:
                current_gene = gene_id

            if gene_id != current_gene:
                gene_df = pd.DataFrame(buffer)
                gene_df = gene_df.sort_values(
                    [pos_col, candidate_index_col],
                    kind="mergesort",
                )
                yield current_gene, gene_df

                current_gene = gene_id
                buffer = []

            buffer.append(d)

    if buffer:
        gene_df = pd.DataFrame(buffer)
        gene_df = gene_df.sort_values(
            [pos_col, candidate_index_col],
            kind="mergesort",
        )
        yield current_gene, gene_df


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else math.nan

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0

    return {
        "n": int(len(y_true)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "mcc": mcc,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input-tsv", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-tsv", required=True)

    ap.add_argument("--window-col", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--group-col", default=None)
    ap.add_argument("--seqid-col", default=None)
    ap.add_argument("--split-col", default=None)
    ap.add_argument("--pos-col", default=None)
    ap.add_argument("--candidate-type-col", default=None)
    ap.add_argument("--candidate-index-col", default="candidate_index")

    ap.add_argument("--chunksize", type=int, default=100_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--max-genes", type=int, default=None)

    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    input_tsv = Path(args.input_tsv)
    out_tsv = Path(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoint:", checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", ckpt.get("config", {}))
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))

    window_col = args.window_col or ckpt_args.get("window_col", "window_seq_600bp")
    label_col = args.label_col or ckpt_args.get("label_col", "curated_is_used")
    group_col = args.group_col or ckpt_args.get("group_col", "gene_id")
    seqid_col = args.seqid_col or ckpt_args.get("seqid_col", "seqid")
    split_col = args.split_col or ckpt_args.get("split_col", "split")
    pos_col = args.pos_col or ckpt_args.get("pos_col", "relative_base_pos")
    candidate_type_col = args.candidate_type_col or ckpt_args.get("candidate_type_col", "candidate_type")
    candidate_index_col = args.candidate_index_col

    print("Using columns:")
    print(json.dumps({
        "window_col": window_col,
        "label_col": label_col,
        "group_col": group_col,
        "seqid_col": seqid_col,
        "split_col": split_col,
        "pos_col": pos_col,
        "candidate_type_col": candidate_type_col,
        "candidate_index_col": candidate_index_col,
    }, indent=2))

    header = read_header(input_tsv)

    required = [
        window_col,
        group_col,
        seqid_col,
        split_col,
        pos_col,
        candidate_type_col,
        candidate_index_col,
    ]

    if label_col is not None and label_col in header:
        required.append(label_col)

    optional = [
        "source_row",
        "type_index",
        "motif",
        "strand",
        "relative_fraction",
        "curated_label",
        "start",
        "end",
        "genomic_pos_1based",
        "center_index_600bp",
        "center_base_600bp",
        "center_2bp_600bp",
        "center_6bp_600bp",
    ]

    usecols = []
    for col in required + optional:
        if col in header and col not in usecols:
            usecols.append(col)

    missing = [col for col in required if col not in header]
    if missing:
        raise ValueError(f"Input TSV missing columns: {missing}")

    device = torch.device(args.device)

    model = NTPoolCandidateTransformer(
        model_id=ckpt_args.get("model_id", "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"),
        nt_layer=int(ckpt_args.get("nt_layer", -1)),
        freeze_nt=bool(ckpt_args.get("freeze_nt", False)),
        unfreeze_last_nt_layers=int(ckpt_args.get("unfreeze_last_nt_layers", 0)),
        window_size=int(ckpt_args.get("window_size", 600)),
        num_classes=int(ckpt_args.get("num_classes", 2)),
        transformer_model_dim=int(ckpt_args.get("transformer_model_dim", ckpt_args.get("model_dim", 256))),
        transformer_layers=int(ckpt_args.get("transformer_layers", ckpt_args.get("num_layers", 2))),
        transformer_heads=int(ckpt_args.get("transformer_heads", ckpt_args.get("n_heads", 4))),
        transformer_dropout=float(ckpt_args.get("dropout", 0.1)),
        append_position_features=True,
        pooling_dropout=float(ckpt_args.get("dropout", 0.1)),
    ).to(device)

    missing_state, unexpected_state = model.load_state_dict(state, strict=False)

    print("load_state_dict missing:", missing_state)
    print("load_state_dict unexpected:", unexpected_state)

    model.eval()

    if out_tsv.exists():
        print("Remove existing output:", out_tsv)
        out_tsv.unlink()

    y_true_all = []
    y_pred_all = []
    n_rows = 0
    n_genes = 0
    first_write = True

    autocast_enabled = bool(args.amp and device.type == "cuda")

    with torch.no_grad():
        gene_iter = iter_gene_groups_sorted(
            path=input_tsv,
            usecols=usecols,
            group_col=group_col,
            pos_col=pos_col,
            candidate_index_col=candidate_index_col,
            chunksize=args.chunksize,
        )

        for gene_id, gene_df in tqdm(gene_iter, desc="exporting NT logits by gene"):
            if args.max_genes is not None and n_genes >= args.max_genes:
                break

            windows = gene_df[window_col].astype(str).tolist()
            candidate_types = gene_df[candidate_type_col].astype(str).tolist()
            pos = gene_df[pos_col].astype(float).tolist()

            n = len(windows)

            batch_windows = [windows]
            candidate_mask = torch.ones((1, n), dtype=torch.bool, device=device)
            position_features = build_position_features(pos, candidate_types).unsqueeze(0).to(device)

            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                out = model(
                    batch_windows=batch_windows,
                    candidate_mask=candidate_mask,
                    labels=None,
                    position_features=position_features,
                )
                logits = out["logits"]

            logits_cpu = logits.detach().float().cpu()[0, :n]
            probs_cpu = torch.softmax(logits_cpu, dim=-1)
            preds_cpu = probs_cpu.argmax(dim=-1)

            out_df = gene_df.drop(columns=[window_col]).copy()

            out_df["logit_not_used"] = logits_cpu[:, 0].numpy()

            if logits_cpu.shape[-1] > 1:
                out_df["logit_used"] = logits_cpu[:, 1].numpy()
                out_df["logit_margin_used_vs_not"] = (
                    out_df["logit_used"] - out_df["logit_not_used"]
                )
                out_df["prob_not_used"] = probs_cpu[:, 0].numpy()
                out_df["prob_used"] = probs_cpu[:, 1].numpy()
            else:
                out_df["logit_used"] = np.nan
                out_df["logit_margin_used_vs_not"] = np.nan
                out_df["prob_not_used"] = probs_cpu[:, 0].numpy()
                out_df["prob_used"] = np.nan

            out_df["score"] = out_df["prob_used"]
            out_df["pred_label"] = preds_cpu.numpy().astype(int)

            if label_col in out_df.columns:
                labels = out_df[label_col].astype(int).to_numpy()
                y_true_all.extend(labels.tolist())
                y_pred_all.extend(out_df["pred_label"].astype(int).tolist())

            out_df.to_csv(
                out_tsv,
                sep="\t",
                index=False,
                mode="w" if first_write else "a",
                header=first_write,
            )

            first_write = False
            n_rows += len(out_df)
            n_genes += 1

    print("Wrote:", out_tsv)
    print("Rows written:", n_rows)
    print("Genes processed:", n_genes)

    metrics = {
        "checkpoint": str(checkpoint),
        "input_tsv": str(input_tsv),
        "out_tsv": str(out_tsv),
        "n_rows": int(n_rows),
        "n_genes": int(n_genes),
    }

    if y_true_all:
        metrics.update(compute_metrics(y_true_all, y_pred_all))

    metrics_path = out_tsv.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("Wrote:", metrics_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
