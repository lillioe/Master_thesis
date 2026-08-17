from __future__ import annotations

import argparse
import json
import shutil
import math
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.models.nt_pool_transformer import NTPoolCandidateTransformer

"""
Train the NT pooled candidate Transformer for splice site candidate classification.
Candidate-centered 600 bp windows are encoded by NT, pooled into candidate representations 
and passed through a candidate Transformer. NT backbone can be frozen or partially fine-tuned.
"""

def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, sep="\t", nrows=0).columns.tolist()


class GeneWindowDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def build_position_features(pos, candidate_types):
    """encode relative candidate position, donor/acceptor identity
    Returns norm position, donor and acceptor indicator"""
    pos = torch.tensor(pos, dtype=torch.float32)

    if len(pos) > 1 and float(pos.max() - pos.min()) > 0:
        pos_norm = (pos - pos.min()) / (pos.max() - pos.min())
    else:
        pos_norm = torch.zeros_like(pos)

    is_donor = torch.tensor([1.0 if t == "donor" else 0.0 for t in candidate_types], dtype=torch.float32)
    is_acceptor = torch.tensor([1.0 if t == "acceptor" else 0.0 for t in candidate_types], dtype=torch.float32)

    return torch.stack([pos_norm, is_donor, is_acceptor], dim=1)


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
            }
        )

    return {
        "batch_windows": batch_windows,
        "candidate_mask": candidate_mask,
        "labels": labels,
        "position_features": position_features,
        "meta": meta,
    }


def decide_fold(row, split_mode, cv_splits, val_chrom):
    split = str(row["split"])
    seqid = str(row["seqid"])

    if split_mode == "fixed":
        if split == "train":
            return "train"
        if split == "val":
            return "val"
        return None

    if split_mode == "leave_one_chrom":
        if split not in cv_splits:
            return None
        if seqid == val_chrom:
            return "val"
        return "train"

    raise ValueError(f"Unknown split_mode: {split_mode}")


def load_gene_examples_from_tsv(
    tsv: Path,
    window_col: str,
    label_col: str,
    group_col: str,
    seqid_col: str,
    split_col: str,
    pos_col: str,
    candidate_type_col: str,
    split_mode: str,
    cv_splits: set[str],
    val_chrom: str | None,
    max_genes_per_split: int,
    max_candidates_per_gene: int,
    chunksize: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """ load candidate windows, group them into ordered seqence.
    Cand. are retrained up to max_candidates_per_gene 
    + ordered by genomic positioin"""
    header = read_header(tsv)

    required = [
        window_col,
        label_col,
        group_col,
        seqid_col,
        split_col,
        pos_col,
        candidate_type_col,
    ]

    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(f"Input TSV missing columns: {missing}")

    usecols = required

    records = {
        "train": {},
        "val": {},
    }

    rng = random.Random(seed)

    reader = pd.read_csv(tsv, sep="\t", usecols=usecols, chunksize=chunksize)

    for chunk_i, chunk in enumerate(reader):
        chunk = chunk.rename(
            columns={
                group_col: "gene_id",
                seqid_col: "seqid",
                split_col: "split",
                pos_col: "pos",
                candidate_type_col: "candidate_type",
            }
        )

        for row in chunk.itertuples(index=False):
            d = row._asdict()

            fold = decide_fold(
                d,
                split_mode=split_mode,
                cv_splits=cv_splits,
                val_chrom=val_chrom,
            )

            if fold is None:
                continue

            gene_id = str(d["gene_id"])
            key = gene_id

            if key not in records[fold]:
                if len(records[fold]) >= max_genes_per_split:
                    continue

                records[fold][key] = {
                    "gene_id": gene_id,
                    "seqid": str(d["seqid"]),
                    "split": str(d["split"]),
                    "windows": [],
                    "labels": [],
                    "pos": [],
                    "candidate_types": [],
                }

            rec = records[fold][key]

            if len(rec["windows"]) >= max_candidates_per_gene:
                continue

            rec["windows"].append(str(getattr(row, window_col)))
            rec["labels"].append(int(getattr(row, label_col)))
            rec["pos"].append(float(d["pos"]))
            rec["candidate_types"].append(str(d["candidate_type"]))

        if chunk_i % 10 == 0:
            print(
                f"chunk {chunk_i:,}; "
                f"train genes={len(records['train']):,}; "
                f"val genes={len(records['val']):,}"
            )

        if (
            len(records["train"]) >= max_genes_per_split
            and len(records["val"]) >= max_genes_per_split
        ):
            break

    def materialize(fold):
        examples = []

        for rec in records[fold].values():
            order = sorted(range(len(rec["windows"])), key=lambda i: rec["pos"][i])

            windows = [rec["windows"][i] for i in order]
            labels = [rec["labels"][i] for i in order]
            pos = [rec["pos"][i] for i in order]
            candidate_types = [rec["candidate_types"][i] for i in order]

            if len(set(labels)) < 2:
                # Keep gene anyway, class balance is handled globally
                pass

            examples.append(
                {
                    "gene_id": rec["gene_id"],
                    "seqid": rec["seqid"],
                    "split": rec["split"],
                    "windows": windows,
                    "labels": labels,
                    "position_features": build_position_features(pos, candidate_types),
                }
            )

        rng.shuffle(examples)
        return examples

    train_examples = materialize("train")
    val_examples = materialize("val")

    if not train_examples:
        raise ValueError("No train examples collected")
    if not val_examples:
        raise ValueError("No val examples collected")

    return train_examples, val_examples


def count_labels(examples: list[dict], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)

    for e in examples:
        y = torch.tensor(e["labels"], dtype=torch.long)
        for c in range(num_classes):
            counts[c] += int((y == c).sum().item())

    return counts


def make_class_weights(counts: torch.Tensor) -> torch.Tensor:
    total = counts.sum().float()
    return total / (len(counts) * counts.float().clamp(min=1))


def run_one_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer=None,
    max_batches=None,
    grad_clip=1.0,
):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_n = 0
    tp = fp = tn = fn = 0

    iterator = tqdm(loader, desc="train" if is_train else "val", leave=False)

    for batch_i, batch in enumerate(iterator):
        if max_batches is not None and batch_i >= max_batches:
            break

        candidate_mask = batch["candidate_mask"].to(device)
        labels = batch["labels"].to(device)
        position_features = batch["position_features"].to(device)

        with torch.set_grad_enabled(is_train):
            out = model(
                batch_windows=batch["batch_windows"],
                candidate_mask=candidate_mask,
                labels=None,
                position_features=position_features,
            )

            logits = out["logits"]

            loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

        valid = labels != -100
        preds = logits.argmax(dim=-1)

        y_true = labels[valid]
        y_pred = preds[valid]

        n = int(valid.sum().item())
        total_n += n
        total_loss += float(loss.item()) * n

        tp += int(((y_true == 1) & (y_pred == 1)).sum().item())
        fp += int(((y_true == 0) & (y_pred == 1)).sum().item())
        tn += int(((y_true == 0) & (y_pred == 0)).sum().item())
        fn += int(((y_true == 1) & (y_pred == 0)).sum().item())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "loss": total_loss / total_n if total_n else math.nan,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_candidates": total_n,
    }


def save_checkpoint(path, model, optimizer, epoch, args, val_metrics, best_val_f1):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "val_metrics": val_metrics,
            "best_val_f1": best_val_f1,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-tsv", required=True)
    parser.add_argument("--window-col", default="window_seq_600bp")
    parser.add_argument("--label-col", default="curated_is_used")
    parser.add_argument("--group-col", default="gene_id")
    parser.add_argument("--seqid-col", default="seqid")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--pos-col", default="relative_base_pos")
    parser.add_argument("--candidate-type-col", default="candidate_type")

    parser.add_argument("--model-id", default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
    parser.add_argument("--nt-layer", type=int, default=-1)
    parser.add_argument("--freeze-nt", action="store_true")
    parser.add_argument("--unfreeze-last-nt-layers", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=600)

    parser.add_argument("--split-mode", choices=["fixed", "leave_one_chrom"], default="leave_one_chrom")
    parser.add_argument("--cv-splits", nargs="+", default=["train", "val"])
    parser.add_argument("--val-chrom", default="NC_000019.10")

    parser.add_argument("--max-genes-per-split", type=int, default=100)
    parser.add_argument("--max-candidates-per-gene", type=int, default=256)
    parser.add_argument("--chunksize", type=int, default=100000)

    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--transformer-model-dim", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--nt-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--max-batches-train", type=int, default=None)
    parser.add_argument("--max-batches-val", type=int, default=None)

    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--override-lr-on-resume", action="store_true")
    parser.add_argument("--save-every-epoch", action="store_true", help="Save full epoch checkpoints as epoch_XXX.pt")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading gene-window examples")
    train_examples, val_examples = load_gene_examples_from_tsv(
        tsv=Path(args.input_tsv),
        window_col=args.window_col,
        label_col=args.label_col,
        group_col=args.group_col,
        seqid_col=args.seqid_col,
        split_col=args.split_col,
        pos_col=args.pos_col,
        candidate_type_col=args.candidate_type_col,
        split_mode=args.split_mode,
        cv_splits=set(args.cv_splits),
        val_chrom=args.val_chrom,
        max_genes_per_split=args.max_genes_per_split,
        max_candidates_per_gene=args.max_candidates_per_gene,
        chunksize=args.chunksize,
        seed=args.seed,
    )

    print(f"Train genes: {len(train_examples):,}")
    print(f"Val genes:   {len(val_examples):,}")

    train_counts = count_labels(train_examples, args.num_classes)
    val_counts = count_labels(val_examples, args.num_classes)

    print(f"Train label counts: {train_counts.tolist()}")
    print(f"Val label counts:   {val_counts.tolist()}")

    train_loader = DataLoader(
        GeneWindowDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_gene_windows,
    )

    val_loader = DataLoader(
        GeneWindowDataset(val_examples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_gene_windows,
    )

    device = torch.device(args.device)

    model = NTPoolCandidateTransformer(
        model_id=args.model_id,
        nt_layer=args.nt_layer,
        freeze_nt=args.freeze_nt,
        unfreeze_last_nt_layers=args.unfreeze_last_nt_layers,
        window_size=args.window_size,
        num_classes=args.num_classes,
        transformer_model_dim=args.transformer_model_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_dropout=args.dropout,
        append_position_features=True,
        pooling_dropout=args.dropout,
    ).to(device)

    if args.class_weights:
        weights = make_class_weights(train_counts).to(device)
        print(f"Class weights: {weights.detach().cpu().tolist()}")
    else:
        weights = None

    criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)

    nt_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("nt."):
            nt_params.append(p)
        else:
            other_params.append(p)

    param_groups = [
        {"params": other_params, "lr": args.lr},
    ]

    if nt_params:
        param_groups.append({"params": nt_params, "lr": args.nt_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    start_epoch = 1
    best_val_f1 = -1.0
    metrics_rows = []

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if args.override_lr_on_resume:
            optimizer.param_groups[0]["lr"] = args.lr
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]["lr"] = args.nt_lr

            print("Overrode optimizer learning rates:")
            for i, group in enumerate(optimizer.param_groups):
                print(f"  param_group {i}: lr={group['lr']}")

        start_epoch = int(ckpt["epoch"]) + 1
        best_val_f1 = float(ckpt.get("best_val_f1", -1.0))
        print(f"Resumed from {args.resume_from}; starting at epoch {start_epoch}")

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            max_batches=args.max_batches_train,
            grad_clip=args.grad_clip,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            optimizer=None,
            max_batches=args.max_batches_val,
            grad_clip=None,
        )

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        metrics_rows.append(row)

        pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics.tsv", sep="\t", index=False)

        print(
            f"train loss={train_metrics['loss']:.4f} "
            f"f1={train_metrics['f1']:.4f} "
            f"prec={train_metrics['precision']:.4f} "
            f"rec={train_metrics['recall']:.4f}"
        )

        print(
            f"val   loss={val_metrics['loss']:.4f} "
            f"f1={val_metrics['f1']:.4f} "
            f"prec={val_metrics['precision']:.4f} "
            f"rec={val_metrics['recall']:.4f}"
        )

        save_checkpoint(
            out_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            val_metrics=val_metrics,
            best_val_f1=best_val_f1,
        )

        if args.save_every_epoch:
            epoch_ckpt_path = out_dir / f"epoch_{epoch:03d}.pt"
            shutil.copy2(out_dir / "last.pt", epoch_ckpt_path)
            print(f"Saved epoch checkpoint to {epoch_ckpt_path}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_checkpoint(
                out_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                val_metrics=val_metrics,
                best_val_f1=best_val_f1,
            )
            print(f"Saved new best checkpoint with val_f1={best_val_f1:.4f}")

    print(f"Best val F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()