import pandas as pd

GFF_COLUMNS = [
    "seqid", "source", "type",
    "start", "end", "score",
    "strand", "phase", "attributes",
]

def load_gff(gff_path: str) -> pd.DataFrame:
    return pd.read_csv(
        gff_path,
        sep="\t",
        comment="#",
        header=None,
        names=GFF_COLUMNS,
        compression="infer",
    )

def get_gff_attr(attr: str, key: str):
    if pd.isna(attr):
        return None

    for item in attr.split(";"):
        if item.startswith(key + "="):
            return item.split("=", 1)[1]

    return None

def build_genes_df(gff: pd.DataFrame) -> pd.DataFrame:
    genes = gff[gff["type"] == "gene"].copy()

    genes["gene_id"] = genes["attributes"].apply(lambda x: get_gff_attr(x, "ID"))
    genes["gene_name"] = genes["attributes"].apply(lambda x: get_gff_attr(x, "gene"))
    genes["gene_name_alt"] = genes["attributes"].apply(lambda x: get_gff_attr(x, "Name"))
    genes["gene_name"] = genes["gene_name"].fillna(genes["gene_name_alt"])

    genes_df = genes[
        ["gene_id", "seqid", "start", "end", "strand", "gene_name"]
    ].copy()

    genes_df = genes_df.dropna(subset=["gene_id"])
    genes_df = genes_df[genes_df["strand"].isin(["+", "-"])]
    genes_df = genes_df.drop_duplicates().reset_index(drop=True)

    return genes_df

def build_transcripts_df(gff: pd.DataFrame) -> pd.DataFrame:
    """
    Build transcript table from GFF3
    """

    transcript_types = {
        "mRNA",
        "transcript",
        "lnc_RNA",
        "ncRNA",
        "rRNA",
        "tRNA",
        "snoRNA",
        "snRNA",
        "miRNA",
        "primary_transcript",
    }

    tx = gff[gff["type"].isin(transcript_types)].copy()

    tx["transcript_id"] = tx["attributes"].apply(lambda x: get_gff_attr(x, "ID"))
    tx["gene_id"] = tx["attributes"].apply(lambda x: get_gff_attr(x, "Parent"))
    tx["transcript_type"] = tx["type"]

    transcripts_df = tx[
        [
            "gene_id",
            "transcript_id",
            "seqid",
            "start",
            "end",
            "strand",
            "transcript_type",
        ]
    ].copy()

    transcripts_df = transcripts_df.dropna(subset=["gene_id", "transcript_id"])
    transcripts_df = transcripts_df[transcripts_df["strand"].isin(["+", "-"])]
    transcripts_df = transcripts_df.drop_duplicates().reset_index(drop=True)

    return transcripts_df

def build_exons_df(
    gff: pd.DataFrame,
    transcripts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build exon table and attach gene_id through transcript_id

    In GFF3, exon rows usually have:
        Parent=rna-...
    Sometimes Parent can contain multiple transcript IDs separated by commas
    """

    exons = gff[gff["type"] == "exon"].copy()

    exons["transcript_id"] = exons["attributes"].apply(
        lambda x: get_gff_attr(x, "Parent")
    )

    # Handle cases where one exon has multiple transcript parents.
    exons["transcript_id"] = exons["transcript_id"].str.split(",")
    exons = exons.explode("transcript_id")

    exons_df = exons[
        [
            "transcript_id",
            "seqid",
            "start",
            "end",
            "strand",
        ]
    ].copy()

    exons_df = exons_df.dropna(subset=["transcript_id"])
    exons_df = exons_df[exons_df["strand"].isin(["+", "-"])]

    # Add gene_id using transcript_id → gene_id mapping.
    exons_df = exons_df.merge(
        transcripts_df[["gene_id", "transcript_id"]],
        on="transcript_id",
        how="inner",
    )

    exons_df = exons_df[
        [
            "gene_id",
            "transcript_id",
            "seqid",
            "start",
            "end",
            "strand",
        ]
    ].drop_duplicates().reset_index(drop=True)

    return exons_df

def filter_to_multi_exon(genes_df, transcripts_df, exons_df):
    """
    Keep only transcripts with at least two exons, plus their genes and exons
    """
    exon_counts = (
        exons_df
        .groupby(["gene_id", "transcript_id"])
        .size()
        .reset_index(name="n_exons")
    )

    multi_exon = exon_counts[exon_counts["n_exons"] >= 2][
        ["gene_id", "transcript_id"]
    ]

    transcripts_df = transcripts_df.merge(
        multi_exon,
        on=["gene_id", "transcript_id"],
        how="inner",
    )

    exons_df = exons_df.merge(
        multi_exon,
        on=["gene_id", "transcript_id"],
        how="inner",
    )

    genes_df = genes_df[
        genes_df["gene_id"].isin(transcripts_df["gene_id"])
    ].reset_index(drop=True)

    return genes_df, transcripts_df, exons_df
