#!/usr/bin/env python3
"""
Train a splice-path CRF on top of exported NT-transformer logits

Inputs:
    nt_logits.tsv
        One row per candidate, containing:
            gene_id
            candidate_index
            candidate_type
            logit_not_used
            logit_used

    transcript_paths_valid.tsv
        One row per valid transcript path, containing:
            gene_id
            transcript_id
            selected_candidate_indices
            selected_candidate_types

    selected_crf_genes.tsv
        One row per selected gene, containing:
            gene_id
            split

Training objective:
    Negative CRF log-likelihood of annotated transcript-specific paths
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.splice_path_crf import (
    LABELS_3,
    LABELS_4,
    SpliceCRFConfig,
    SplicePathCRFModel,
)


DONOR_CANDIDATE = 0
ACCEPTOR_CANDIDATE = 1


@dataclass
class GeneData:
    gene_id: str
    candidate_indices: List[int]
    site_logits: torch.Tensor          # (L, 2)
    candidate_type_ids: torch.Tensor   # (L,)


@dataclass
class TranscriptExample:
    gene_id: str
    transcript_id: str
    selected: Dict[int, str]           # candidate_index to donor/acceptor


def parse_semicolon_list(value) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text == "":
        return []
    return [x for x in text.split(";") if x != ""]


def parse_selected_path(row: pd.Series) -> Dict[int, str]:
    idxs = parse_semicolon_list(row["selected_candidate_indices"])
    types = parse_semicolon_list(row["selected_candidate_types"])

    if len(idxs) != len(types):
        raise ValueError(
            f"for transcript {row.get('transcript_id')}"
        )

    selected = {}
    for idx, typ in zip(idxs, types):
        typ = typ.lower()
        if typ not in {"donor", "acceptor"}:
            raise ValueError(f"Unexpected: {typ}")
        selected[int(idx)] = typ

    return selected


def candidate_type_to_id(value) -> int:
    """Map donor and acceptor candidate types to integer IDs."""
    text = str(value).lower()
    if text == "donor":
        return DONOR_CANDIDATE
    if text == "acceptor":
        return ACCEPTOR_CANDIDATE


def load_gene_data(nt_logits_path: Path) -> Dict[str, GeneData]:
    """Load candidate NT logits, organize them into ordered gene-specific sequences"""
    df = pd.read_csv(nt_logits_path, sep="\t")

    required = {
        "gene_id",
        "candidate_index",
        "candidate_type",
        "logit_not_used",
        "logit_used",
    }
    missing = required - set(df.columns)

    before = len(df)
    df = df.drop_duplicates(["gene_id", "candidate_index"])
    after = len(df)

    df["candidate_index"] = df["candidate_index"].astype(int)
    df = df.sort_values(["gene_id", "candidate_index"], kind="mergesort")

    gene_data = {}
    for gene_id, sub in df.groupby("gene_id", sort=False):
        candidate_indices = sub["candidate_index"].astype(int).tolist()

        site_logits = torch.tensor(
            sub[["logit_not_used", "logit_used"]].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        candidate_type_ids = torch.tensor(
            [candidate_type_to_id(x) for x in sub["candidate_type"]],
            dtype=torch.long,
        )

        gene_data[str(gene_id)] = GeneData(
            gene_id=str(gene_id),
            candidate_indices=candidate_indices,
            site_logits=site_logits,
            candidate_type_ids=candidate_type_ids,
        )

    return gene_data


def load_split_map(selected_genes_path: Path, nt_logits_path: Path) -> Dict[str, str]:
    """Load the train/val split"""
    selected = pd.read_csv(selected_genes_path, sep="\t")

    split_col = None
    for col in selected.columns:
        if col.lower() == "split":
            split_col = col
            break

    if split_col is not None:
        split_map = {
            str(row["gene_id"]): str(row[split_col]).lower()
            for _, row in selected.iterrows()
        }
        return split_map

    logits_cols = pd.read_csv(nt_logits_path, sep="\t", nrows=5).columns
    logits = pd.read_csv(nt_logits_path, sep="\t", usecols=["gene_id", "split"])
    logits = logits.drop_duplicates(["gene_id"])
    split_map = {
        str(row["gene_id"]): str(row["split"]).lower()
        for _, row in logits.iterrows()
    }
    return split_map


def normalize_split_name(split: str) -> str:
    """norm split names"""
    split = str(split).lower()
    if split in {"train", "training"}:
        return "train"
    if split in {"val", "valid", "validation", "dev"}:
        return "val"
    if "train" in split:
        return "train"
    if "val" in split or "valid" in split or "dev" in split:
        return "val"
    return split


def sample_genes(genes: List[str], max_genes: int | None, seed: int) -> List[str]:
    """random subsample genes for specified number"""
    genes = sorted(set(genes))
    if max_genes is None or max_genes <= 0 or max_genes >= len(genes):
        return genes
    rng = random.Random(seed)
    sampled = rng.sample(genes, max_genes)
    return sorted(sampled)


def load_transcript_examples(
    transcript_paths_path: Path,
    gene_data: Dict[str, GeneData],
    split_map: Dict[str, str],
    max_train_genes: int | None,
    max_val_genes: int | None,
    seed: int,
) -> Tuple[List[TranscriptExample], List[TranscriptExample], List[str], List[str]]:
    """load valid transcript paths, construct training and val examples"""
    paths = pd.read_csv(transcript_paths_path, sep="\t")
    required = {
        "gene_id",
        "transcript_id",
        "selected_candidate_indices",
        "selected_candidate_types",
    }
    missing = required - set(paths.columns)

    if "path_valid" in paths.columns:
        paths = paths[paths["path_valid"] == 1].copy()

    paths["gene_id"] = paths["gene_id"].astype(str)
    paths["transcript_id"] = paths["transcript_id"].astype(str)

    available_genes = set(gene_data)
    paths = paths[paths["gene_id"].isin(available_genes)].copy()
    paths = paths[paths["gene_id"].isin(split_map)].copy()

    train_genes_all = []
    val_genes_all = []

    for gene_id, split in split_map.items():
        split_norm = normalize_split_name(split)
        if gene_id not in available_genes:
            continue
        if split_norm == "train":
            train_genes_all.append(gene_id)
        elif split_norm == "val":
            val_genes_all.append(gene_id)

    train_genes = sample_genes(train_genes_all, max_train_genes, seed)
    val_genes = sample_genes(val_genes_all, max_val_genes, seed + 1)

    train_gene_set = set(train_genes)
    val_gene_set = set(val_genes)

    train_rows = paths[paths["gene_id"].isin(train_gene_set)]
    val_rows = paths[paths["gene_id"].isin(val_gene_set)]

    def rows_to_examples(df: pd.DataFrame) -> List[TranscriptExample]:
        examples = []
        for _, row in df.iterrows():
            selected = parse_selected_path(row)
            examples.append(
                TranscriptExample(
                    gene_id=str(row["gene_id"]),
                    transcript_id=str(row["transcript_id"]),
                    selected=selected,
                )
            )
        return examples

    train_examples = rows_to_examples(train_rows)
    val_examples = rows_to_examples(val_rows)

    return train_examples, val_examples, train_genes, val_genes

# construct 3-label, 4-label CRF state sequence
def build_labels_for_example(
    gene: GeneData,
    example: TranscriptExample,
    label_mode: str,
) -> torch.Tensor:
    candidate_indices = gene.candidate_indices
    selected = example.selected

    if label_mode == "3":
        labels = torch.full((len(candidate_indices),), LABELS_3["skip"], dtype=torch.long)

        for i, candidate_index in enumerate(candidate_indices):
            typ = selected.get(candidate_index)
            if typ == "donor":
                labels[i] = LABELS_3["donor"]
            elif typ == "acceptor":
                labels[i] = LABELS_3["acceptor"]

        return labels

    if label_mode == "4":
        labels = torch.empty((len(candidate_indices),), dtype=torch.long)
        expected_next = "donor"

        for i, candidate_index in enumerate(candidate_indices):
            typ = selected.get(candidate_index)

            if typ == "donor":
                labels[i] = LABELS_4["D"]
                expected_next = "acceptor"
            elif typ == "acceptor":
                labels[i] = LABELS_4["A"]
                expected_next = "donor"
            else:
                if expected_next == "donor":
                    labels[i] = LABELS_4["S_D"]
                else:
                    labels[i] = LABELS_4["S_A"]

        return labels


class SplicePathDataset(Dataset):
    """PyTorch data for transcript specific CRF train examples"""
    def __init__(
        self,
        examples: List[TranscriptExample],
        gene_data: Dict[str, GeneData],
        label_mode: str,
    ):
        self.examples = examples
        self.gene_data = gene_data
        self.label_mode = label_mode

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        gene = self.gene_data[ex.gene_id]

        labels = build_labels_for_example(
            gene=gene,
            example=ex,
            label_mode=self.label_mode,
        )

        return {
            "gene_id": ex.gene_id,
            "transcript_id": ex.transcript_id,
            "candidate_indices": gene.candidate_indices,
            "site_logits": gene.site_logits,
            "candidate_type_ids": gene.candidate_type_ids,
            "labels": labels,
        }


def collate_batch(items: List[dict]) -> dict:
    """combine gene cand sequences into batch"""
    batch_size = len(items)
    max_len = max(item["site_logits"].shape[0] for item in items)

    site_logits = torch.zeros((batch_size, max_len, 2), dtype=torch.float32)
    candidate_type_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -1, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    gene_ids = []
    transcript_ids = []
    candidate_indices = []

    for b, item in enumerate(items):
        L = item["site_logits"].shape[0]

        site_logits[b, :L] = item["site_logits"]
        candidate_type_ids[b, :L] = item["candidate_type_ids"]
        labels[b, :L] = item["labels"]
        attention_mask[b, :L] = True

        gene_ids.append(item["gene_id"])
        transcript_ids.append(item["transcript_id"])
        candidate_indices.append(item["candidate_indices"])

    return {
        "gene_ids": gene_ids,
        "transcript_ids": transcript_ids,
        "candidate_indices": candidate_indices,
        "site_logits": site_logits,
        "candidate_type_ids": candidate_type_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def selected_set_from_labels(
    labels: Iterable[int],
    candidate_indices: List[int],
    label_mode: str,
) -> set:
    """convert decdoed CRF label sequence into set of selected donor, acceptor cand"""
    selected = set()

    if label_mode == "3":
        for lab, idx in zip(labels, candidate_indices):
            lab = int(lab)
            if lab == LABELS_3["donor"]:
                selected.add((int(idx), "donor"))
            elif lab == LABELS_3["acceptor"]:
                selected.add((int(idx), "acceptor"))

    else:
        for lab, idx in zip(labels, candidate_indices):
            lab = int(lab)
            if lab == LABELS_4["D"]:
                selected.add((int(idx), "donor"))
            elif lab == LABELS_4["A"]:
                selected.add((int(idx), "acceptor"))

    return selected


def gold_sets_by_gene(examples: List[TranscriptExample]) -> Dict[str, List[set]]:
    """Group annotated cand sets"""
    out: Dict[str, List[set]] = {}
    for ex in examples:
        selected = {(idx, typ) for idx, typ in ex.selected.items()}
        out.setdefault(ex.gene_id, []).append(selected)
    return out


def prf(pred: set, gold: set) -> Tuple[float, float, float]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def jaccard(pred: set, gold: set) -> float:
    union = pred | gold
    if not union:
        return 1.0
    return len(pred & gold) / len(union)


def micro_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


@torch.no_grad()
def evaluate_loss(
    model: SplicePathCRFModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    """ calc mean CRF neg log-like for eval data"""

    total_loss = 0.0
    total_n = 0

    for batch in loader:
        site_logits = batch["site_logits"].to(device)
        candidate_type_ids = batch["candidate_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(
            site_logits=site_logits,
            candidate_type_ids=candidate_type_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        n = labels.shape[0]
        total_loss += float(out["loss"].item()) * n
        total_n += n

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate_viterbi_by_gene(
    model: SplicePathCRFModel,
    gene_data: Dict[str, GeneData],
    val_genes: List[str],
    val_gold_by_gene: Dict[str, List[set]],
    label_mode: str,
    device: torch.device,
    out_predictions_path: Path | None = None,
) -> dict:
    model.eval()
    """decode viterbi path, compare with annot paths"""

    rows = []

    n_genes = 0
    exact_any_count = 0

    best_f1_values = []
    best_jaccard_values = []

    selected_tp = selected_fp = selected_fn = 0
    donor_tp = donor_fp = donor_fn = 0
    acceptor_tp = acceptor_fp = acceptor_fn = 0

    for gene_id in val_genes:
        if gene_id not in gene_data:
            continue
        if gene_id not in val_gold_by_gene:
            continue

        gene = gene_data[gene_id]
        L = len(gene.candidate_indices)

        site_logits = gene.site_logits.unsqueeze(0).to(device)
        candidate_type_ids = gene.candidate_type_ids.unsqueeze(0).to(device)
        attention_mask = torch.ones((1, L), dtype=torch.bool, device=device)

        out = model(
            site_logits=site_logits,
            candidate_type_ids=candidate_type_ids,
            attention_mask=attention_mask,
            labels=None,
            decode_mode="viterbi",
        )

        pred_labels = out["predictions"][0]
        pred_set = selected_set_from_labels(
            pred_labels,
            gene.candidate_indices,
            label_mode=label_mode,
        )

        gold_sets = val_gold_by_gene[gene_id]

        exact_any = any(pred_set == gold for gold in gold_sets)
        exact_any_count += int(exact_any)

        best_gold = None
        best_f1 = -1.0
        best_precision = 0.0
        best_recall = 0.0
        best_j = 0.0

        for gold in gold_sets:
            p, r, f = prf(pred_set, gold)
            j = jaccard(pred_set, gold)
            if f > best_f1 or (f == best_f1 and j > best_j):
                best_f1 = f
                best_precision = p
                best_recall = r
                best_j = j
                best_gold = gold

        if best_gold is None:
            continue

        best_f1_values.append(best_f1)
        best_jaccard_values.append(best_j)

        tp = len(pred_set & best_gold)
        fp = len(pred_set - best_gold)
        fn = len(best_gold - pred_set)

        selected_tp += tp
        selected_fp += fp
        selected_fn += fn

        pred_donor = {x for x in pred_set if x[1] == "donor"}
        gold_donor = {x for x in best_gold if x[1] == "donor"}
        pred_acceptor = {x for x in pred_set if x[1] == "acceptor"}
        gold_acceptor = {x for x in best_gold if x[1] == "acceptor"}

        donor_tp += len(pred_donor & gold_donor)
        donor_fp += len(pred_donor - gold_donor)
        donor_fn += len(gold_donor - pred_donor)

        acceptor_tp += len(pred_acceptor & gold_acceptor)
        acceptor_fp += len(pred_acceptor - gold_acceptor)
        acceptor_fn += len(gold_acceptor - pred_acceptor)

        rows.append(
            {
                "gene_id": gene_id,
                "n_candidates": L,
                "n_gold_transcripts": len(gold_sets),
                "n_pred_selected": len(pred_set),
                "exact_match_any": int(exact_any),
                "best_selected_precision": best_precision,
                "best_selected_recall": best_recall,
                "best_selected_f1": best_f1,
                "best_selected_jaccard": best_j,
                "pred_selected": ";".join(f"{idx}:{typ}" for idx, typ in sorted(pred_set)),
                "best_gold_selected": ";".join(f"{idx}:{typ}" for idx, typ in sorted(best_gold)),
            }
        )

        n_genes += 1

    selected_p, selected_r, selected_f1 = micro_prf(
        selected_tp,
        selected_fp,
        selected_fn,
    )
    donor_p, donor_r, donor_f1 = micro_prf(
        donor_tp,
        donor_fp,
        donor_fn,
    )
    acceptor_p, acceptor_r, acceptor_f1 = micro_prf(
        acceptor_tp,
        acceptor_fp,
        acceptor_fn,
    )

    if out_predictions_path is not None:
        pd.DataFrame(rows).to_csv(out_predictions_path, sep="\t", index=False)

    return {
        "val_exact_path_match_any": exact_any_count / max(n_genes, 1),
        "val_best_selected_f1": float(np.mean(best_f1_values)) if best_f1_values else 0.0,
        "val_best_selected_jaccard": float(np.mean(best_jaccard_values)) if best_jaccard_values else 0.0,
        "val_selected_precision": selected_p,
        "val_selected_recall": selected_r,
        "val_selected_f1": selected_f1,
        "val_donor_precision": donor_p,
        "val_donor_recall": donor_r,
        "val_donor_f1": donor_f1,
        "val_acceptor_precision": acceptor_p,
        "val_acceptor_recall": acceptor_r,
        "val_acceptor_f1": acceptor_f1,
        "val_eval_genes": n_genes,
    }


def train_one_epoch(
    model: SplicePathCRFModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()

    total_loss = 0.0
    total_n = 0

    for batch in loader:
        site_logits = batch["site_logits"].to(device)
        candidate_type_ids = batch["candidate_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(
            site_logits=site_logits,
            candidate_type_ids=candidate_type_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = out["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        n = labels.shape[0]
        total_loss += float(loss.item()) * n
        total_n += n

    return total_loss / max(total_n, 1)


def save_checkpoint(
    path: Path,
    model: SplicePathCRFModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    args: argparse.Namespace,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "args": vars(args),
        "label_mode": args.label_mode,
        "use_logit_calibration": not args.no_logit_calibration,
        "enforce_transition_constraints": args.enforce_transition_constraints,
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--nt-logits", required=True, type=Path)
    parser.add_argument("--transcript-paths", required=True, type=Path)
    parser.add_argument("--selected-genes", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)

    parser.add_argument("--label-mode", choices=["3", "4"], required=True)
    parser.add_argument("--enforce-transition-constraints", action="store_true")
    parser.add_argument("--no-logit-calibration", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--max-train-genes", type=int, default=None)
    parser.add_argument("--max-val-genes", type=int, default=None)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--reset-optimizer-on-resume", action="store_true")
    parser.add_argument("--save-all-checkpoints", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gene_data = load_gene_data(args.nt_logits)

    split_map = load_split_map(args.selected_genes, args.nt_logits)
    train_examples, val_examples, train_genes, val_genes = load_transcript_examples(
        transcript_paths_path=args.transcript_paths,
        gene_data=gene_data,
        split_map=split_map,
        max_train_genes=args.max_train_genes,
        max_val_genes=args.max_val_genes,
        seed=args.seed,
    )

    print(f"Genes with logits: {len(gene_data)}")
    print(f"Train genes: {len(train_genes)}")
    print(f"Val genes: {len(val_genes)}")
    print(f"Train transcript examples: {len(train_examples)}")
    print(f"Val transcript examples: {len(val_examples)}")

    if len(train_examples) == 0:
        raise RuntimeError("No training examples found.")
    if len(val_examples) == 0:
        raise RuntimeError("No validation examples found.")

    train_dataset = SplicePathDataset(
        examples=train_examples,
        gene_data=gene_data,
        label_mode=args.label_mode,
    )
    val_dataset = SplicePathDataset(
        examples=val_examples,
        gene_data=gene_data,
        label_mode=args.label_mode,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    config = SpliceCRFConfig(
        label_mode=args.label_mode,
        use_logit_calibration=not args.no_logit_calibration,
        enforce_transition_constraints=args.enforce_transition_constraints,
    )

    model = SplicePathCRFModel(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0

    if args.resume_checkpoint is not None:
        print(f"Loading checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])

        if args.reset_optimizer_on_resume:
            print("Resetting optimizer state on resume.")
        else:
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("Loaded optimizer state from checkpoint.")
            else:
                print("[warning] checkpoint has no optimizer_state_dict; optimizer starts fresh.")

        start_epoch = int(checkpoint.get("epoch", 0))
        print(f"Resuming after epoch {start_epoch}; next epoch will be {start_epoch + 1}")

        torch.save(checkpoint, args.out_dir / "checkpoint_resume_start.pt")

    val_gold_by_gene = gold_sets_by_gene(val_examples)

    config_json = {
        "nt_logits": str(args.nt_logits),
        "transcript_paths": str(args.transcript_paths),
        "selected_genes": str(args.selected_genes),
        "out_dir": str(args.out_dir),
        "label_mode": args.label_mode,
        "enforce_transition_constraints": args.enforce_transition_constraints,
        "use_logit_calibration": not args.no_logit_calibration,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "patience": args.patience,
        "max_train_genes": args.max_train_genes,
        "max_val_genes": args.max_val_genes,
        "seed": args.seed,
        "device": str(device),
        "n_train_genes": len(train_genes),
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
        "reset_optimizer_on_resume": args.reset_optimizer_on_resume,
        "save_all_checkpoints": args.save_all_checkpoints,
        "n_val_genes": len(val_genes),
        "n_train_examples": len(train_examples),
        "n_val_examples": len(val_examples),
    }

    with open(args.out_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2)

    metrics_rows = []
    best_val_loss = math.inf
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )

        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            device=device,
        )

        pred_path = args.out_dir / f"val_predictions_epoch_{epoch:03d}.tsv"
        viterbi_metrics = evaluate_viterbi_by_gene(
            model=model,
            gene_data=gene_data,
            val_genes=val_genes,
            val_gold_by_gene=val_gold_by_gene,
            label_mode=args.label_mode,
            device=device,
            out_predictions_path=pred_path,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": args.lr,
            **viterbi_metrics,
        }

        metrics_rows.append(row)

        pd.DataFrame(metrics_rows).to_csv(
            args.out_dir / "train_metrics.tsv",
            sep="\t",
            index=False,
        )

        save_checkpoint(
            path=args.out_dir / "checkpoint_last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=row,
            args=args,
        )

        if args.save_all_checkpoints:
            save_checkpoint(
                path=args.out_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=row,
                args=args,
            )

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0

            save_checkpoint(
                path=args.out_dir / "checkpoint_best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=row,
                args=args,
            )

            final_pred_path = args.out_dir / "val_predictions_best.tsv"
            pred_path.replace(final_pred_path)
        else:
            bad_epochs += 1

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"best_val_loss={best_val_loss:.6f} "
            f"best_epoch={best_epoch} "
            f"exact_any={viterbi_metrics['val_exact_path_match_any']:.4f} "
            f"best_f1={viterbi_metrics['val_best_selected_f1']:.4f} "
            f"selected_f1={viterbi_metrics['val_selected_f1']:.4f}"
        )

        if args.patience and bad_epochs >= args.patience:
            print(f"Early stopping after {bad_epochs}")
            break

    print("Done.")
    print("Best epoch:", best_epoch)
    print("Best val_loss:", best_val_loss)
    print("Saved:", args.out_dir)


if __name__ == "__main__":
    main()
