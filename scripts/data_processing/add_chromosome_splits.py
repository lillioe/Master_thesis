from pathlib import Path
import argparse
import pandas as pd


REFSEQ_TO_CHROM = {
    "NC_000001.11": "1",
    "NC_000002.12": "2",
    "NC_000003.12": "3",
    "NC_000004.12": "4",
    "NC_000005.10": "5",
    "NC_000006.12": "6",
    "NC_000007.14": "7",
    "NC_000008.11": "8",
    "NC_000009.12": "9",
    "NC_000010.11": "10",
    "NC_000011.10": "11",
    "NC_000012.12": "12",
    "NC_000013.11": "13",
    "NC_000014.9": "14",
    "NC_000015.10": "15",
    "NC_000016.10": "16",
    "NC_000017.11": "17",
    "NC_000018.10": "18",
    "NC_000019.10": "19",
    "NC_000020.11": "20",
    "NC_000021.9": "21",
    "NC_000022.11": "22",
    "NC_000023.11": "X",
    "NC_000024.10": "Y",
}


def normalize_chrom(x: str) -> str:
    x = str(x).strip()

    if x in REFSEQ_TO_CHROM:
        return REFSEQ_TO_CHROM[x]

    if x.startswith("chr"):
        return x[3:]

    return x


def assign_split(seqid: str) -> str:
    chrom = normalize_chrom(seqid)

    train_chroms = {str(i) for i in range(1, 17)}
    val_chroms = {"17", "18", "19"}
    test_chroms = {"20", "21", "22", "X"}

    if chrom in train_chroms:
        return "train"
    if chrom in val_chroms:
        return "val"
    if chrom in test_chroms:
        return "test"

    return "ignore"


def main():
    parser = argparse.ArgumentParser(
        description="Add chromosome-based split column to large candidate tables"
    )

    parser.add_argument("--infile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seqid-col", default="seqid")
    parser.add_argument("--chunksize", type=int, default=1_000_000)

    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first_write = True
    split_counts = {}

    reader = pd.read_csv(
        args.infile,
        sep="\t",
        chunksize=args.chunksize,
    )

    for i, chunk in enumerate(reader):
        if args.seqid_col not in chunk.columns:
            raise ValueError(f"Column not found: {args.seqid_col}")

        chunk["split"] = chunk[args.seqid_col].apply(assign_split)

        counts = chunk["split"].value_counts()
        for split, n in counts.items():
            split_counts[split] = split_counts.get(split, 0) + int(n)

        chunk.to_csv(
            out_path,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )

        first_write = False
        print(f"Processed chunk {i + 1} ({len(chunk):,} rows)")

    print("\nFinal split counts:")
    for split, n in split_counts.items():
        print(f"{split}: {n:,}")


if __name__ == "__main__":
    main()