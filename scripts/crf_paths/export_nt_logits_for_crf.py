from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
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


class GeneWindowLogitExportDataset(Dataset):
    """
    Dataset for exporting NT logits grouped by gene.

    Each item is one complete gene:
        all candidate windows for that gene,
        ordered by relative_base_pos and candidate_index.
    """

    def __init__(
        self,
        tsv: Path,
        window_col: str,
        label_col: str | None,
        group_col: str,
        seqid_col: str,
        split_col: str,
        pos_col: str,
        candidate_type_col: str,
        split_values: set[str] | None,
        gene_list: set[str] | None,
        chunksize: int,
    ):
        header = read_header(tsv)

        required = [
            window_col,
            group_col,
            seqid_col,
            split_col,
            pos_col,
            candidate_type_col,
        ]

        if label_col is not None and label_col in header:
            required.append(label_col)

        optional = [
            "source_row",
            "candidate_index",
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

        records = {}
        kept = 0

        reader = pd.read_csv(tsv, sep="\t", usecols=usecols, chunksize=chunksize)

        for chunk_i, chunk in enumerate(reader):
            if split_values is not None:
                chunk = chunk[chunk[split_col].astype(str).isin(split_values)].copy()

            if gene_list is not None:
                chunk = chunk[chunk[group_col].astype(str).isin(gene_list)].copy()

            if len(chunk) == 0:
                continue

            for row in chunk.itertuples(index=False):
                d = row._asdict()
                gene_id = str(d[group_col])

                if gene_id not in records:
                    records[gene_id] = {
                        "gene_id": gene_id,
                        "seqid": str(d[seqid_col]),
                        "split": str(d[split_col]),
                        "windows": [],
                        "labels": [],
                        "pos": [],
                        "candidate_indices": [],
                        "candidate_types": [],
                        "meta": [],
                    }

                if label_col is not None and label_col in d:
                    label = int(d[label_col])
                else:
                    label = -100

                candidate_index = int(d["candidate_index"]) if "candidate_index" in d else kept

                meta = {}
                for col in usecols:
                    if col != window_col:
                        meta[col] = d[col]

                meta["_input_order"] = kept

                records[gene_id]["windows"].append(str(d[window_col]))
                records[gene_id]["labels"].append(label)
                records[gene_id]["pos"].append(float(d[pos_col]))
                records[gene_id]["candidate_indices"].append(candidate_index)
                records[gene_id]["candidate_types"].append(str(d[candidate_type_col]))
                records[gene_id]["meta"].append(meta)

                kept += 1

            if chunk_i % 10 == 0:
                print(f"chunk {chunk_i:,}; genes={len(records):,}; kept_rows={kept:,}")

        self.examples = []

        for rec in records.values():
            order = sorted(
                range(len(rec["windows"])),
                key=lambda i: (rec["pos"][i], rec["candidate_indices"][i]),
            )

            self.examples.append(
                {
                    "gene_id": rec["gene_id"],
                    "seqid": rec["seqid"],
                    "split": rec["split"],
                    "windows": [rec["windows"][i] for i in order],
                    "labels": [rec["labels"][i] for i in order],
                    "pos": [rec["pos"][i] for i in order],
                    "candidate_types": [rec["candidate_types"][i] for i in order],
                    "meta": [rec["meta"][i] for i in order],
                }
            )

        self.examples = sorted(
            self.examples,
            key=lambda x: (x["split"], x["seqid"], x["gene_id"]),
        )

        print(f"Loaded genes: {len(self.examples):,}")
        print(f"Loaded rows: {sum(len(e['windows']) for e in self.examples):,}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]

        return {
            "gene_id": item["gene_id"],
            "seqid": item["seqid"],
            "split": item["split"],
            "windows": item["windows"],
            "labels": item["labels"],
            "position_features": build_position_features(
                item["pos"],
                item["candidate_types"],
            ),
            "meta": item["meta"],
        }


def collate_gene_windows(batch: list[dict]) -> dict:
    batch_size = len(batch)
    max_len = max(len(item["windows"]) for item in batch)

    candidate_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    position_features = torch.zeros((batch_size, max_len, 3), dtype=torch.float32)

    batch_windows = []
    meta = []

    for i, item in enumerate(batch):
        n = len(item["windows"])

        batch_windows.append(item["windows"] + [""] * (max_len - n))
        candidate_mask[i, :n] = True
        labels[i, :n] = torch.tensor(item["labels"], dtype=torch.long)
        position_features[i, :n] = item["position_features"]

        meta.append(
            {
                "gene_id": item["gene_id"],
                "seqid": item["seqid"],
                "split": item["split"],
                "n": n,
                "rows": item["meta"],
            }
        )

    return {
        "batch_windows": batch_windows,
        "candidate_mask": candidate_mask,
        "labels": labels,
        "position_features": position_features,
        "meta": meta,
    }


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


def load_gene_list(path: str | None) -> set[str] | None:
    if path is None:
        return None

    p = Path(path)
    df = pd.read_csv(p, sep="\t")

    if "gene_id" not in df.columns:
        raise ValueError(f"Gene list file must contain gene_id column: {p}")

    return set(df["gene_id"].astype(str))


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

    ap.add_argument("--split-values", nargs="+", default=None)
    ap.add_argument("--gene-list", default=None)

    ap.add_argument("--chunksize", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")

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

    print("Using columns:")
    print(json.dumps({
        "window_col": window_col,
        "label_col": label_col,
        "group_col": group_col,
        "seqid_col": seqid_col,
        "split_col": split_col,
        "pos_col": pos_col,
        "candidate_type_col": candidate_type_col,
    }, indent=2))

    split_values = set(args.split_values) if args.split_values else None
    gene_list = load_gene_list(args.gene_list)

    if split_values is not None:
        print("Filtering split values:", sorted(split_values))

    if gene_list is not None:
        print("Filtering gene list size:", len(gene_list))

    dataset = GeneWindowLogitExportDataset(
        tsv=input_tsv,
        window_col=window_col,
        label_col=label_col,
        group_col=group_col,
        seqid_col=seqid_col,
        split_col=split_col,
        pos_col=pos_col,
        candidate_type_col=candidate_type_col,
        split_values=split_values,
        gene_list=gene_list,
        chunksize=args.chunksize,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_gene_windows,
    )

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

    missing, unexpected = model.load_state_dict(state, strict=False)

    print("load_state_dict missing:", missing)
    print("load_state_dict unexpected:", unexpected)

    if missing or unexpected:
        print("WARNING: checkpoint did not load strictly. Check architecture args.")

    model.eval()

    rows = []
    y_true_all = []
    y_pred_all = []

    autocast_enabled = bool(args.amp and device.type == "cuda")

    with torch.no_grad():
        for batch in tqdm(loader, desc="exporting NT logits"):
            candidate_mask = batch["candidate_mask"].to(device)
            position_features = batch["position_features"].to(device)

            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                out = model(
                    batch_windows=batch["batch_windows"],
                    candidate_mask=candidate_mask,
                    labels=None,
                    position_features=position_features,
                )

                logits = out["logits"]

            logits_cpu = logits.detach().float().cpu()
            probs_cpu = torch.softmax(logits_cpu, dim=-1)
            preds_cpu = probs_cpu.argmax(dim=-1)
            labels_cpu = batch["labels"]

            for b, m in enumerate(batch["meta"]):
                n = m["n"]

                for j in range(n):
                    row = dict(m["rows"][j])

                    row["logit_not_used"] = float(logits_cpu[b, j, 0])

                    if logits_cpu.shape[-1] > 1:
                        row["logit_used"] = float(logits_cpu[b, j, 1])
                        row["logit_margin_used_vs_not"] = row["logit_used"] - row["logit_not_used"]
                    else:
                        row["logit_used"] = float("nan")
                        row["logit_margin_used_vs_not"] = float("nan")

                    row["prob_not_used"] = float(probs_cpu[b, j, 0])

                    if probs_cpu.shape[-1] > 1:
                        row["prob_used"] = float(probs_cpu[b, j, 1])
                    else:
                        row["prob_used"] = float("nan")

                    row["score"] = row["prob_used"]
                    row["pred_label"] = int(preds_cpu[b, j])

                    rows.append(row)

                    y = int(labels_cpu[b, j])
                    if y != -100:
                        y_true_all.append(y)
                        y_pred_all.append(int(preds_cpu[b, j]))

    out_df = pd.DataFrame(rows)

    if "_input_order" in out_df.columns:
        out_df = (
            out_df
            .sort_values("_input_order", kind="mergesort")
            .drop(columns=["_input_order"])
        )

    out_df.to_csv(out_tsv, sep="\t", index=False)
    print("Wrote:", out_tsv)

    metrics = {
        "checkpoint": str(checkpoint),
        "input_tsv": str(input_tsv),
        "out_tsv": str(out_tsv),
        "n_rows": int(len(out_df)),
        "n_genes": int(len(dataset)),
        "split_values": list(split_values) if split_values else None,
        "gene_list": str(args.gene_list) if args.gene_list else None,
    }

    if y_true_all:
        metrics.update(compute_metrics(y_true_all, y_pred_all))

    metrics_path = out_tsv.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("Wrote:", metrics_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
