import pandas as pd
from data_preparation import prepare_interaction_data


def get_strand_orientation(row: pd.Series) -> int:
    """
    Determine the relative strand orientation between two motifs.

    Parameters
    ----------
    row : pd.Series
        A row containing 'strand1' and 'strand2' columns.

    Returns
    -------
    int
        1 if +/-, 2 if -/+, and 3 otherwise.
    """
    if row.strand1 == "+" and row.strand2 == "-":
        return 1
    if row.strand1 == "-" and row.strand2 == "+":
        return 2
    return 3


def add_interaction_distance_and_strand_features(
    interactions_df: pd.DataFrame,
    motif_features_dir: str,
    cell_lines: list[str]
) -> pd.DataFrame:
    """
    Add motif strand orientation and distance features to the interaction data.

    Parameters
    ----------
    interactions_df : pd.DataFrame
        DataFrame of motif interactions across cell lines.
    motif_features_dir : str
        Path to directory containing motif-level chromatin feature files.
    cell_lines : list[str]
        List of cell line names.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with 'distance' and 'nodes_pos' columns added.
    """
    motif_features = pd.read_csv(f"{motif_features_dir}/GM__motifs_overlap.csv")

    merged = interactions_df.merge(
        motif_features[["sequence_name", "start", "stop", "strand"]],
        left_on=["sequence_name", "start1", "stop1"],
        right_on=["sequence_name", "start", "stop"]
    ).drop(["start", "stop"], axis=1)

    merged = merged.merge(
        motif_features[["sequence_name", "start", "stop", "strand"]],
        left_on=["sequence_name", "start2", "stop2"],
        right_on=["sequence_name", "start", "stop"],
        suffixes=("1", "2")
    ).drop(["start", "stop"], axis=1)

    # Compute distance and keep only positive values
    merged["distance"] = merged["start2"] - merged["stop1"]
    merged = merged.query("distance > 0")

    # Compute strand relationship
    merged["nodes_pos"] = merged.apply(get_strand_orientation, axis=1)

    # Drop PET and NaN-count columns
    drop_cols = [f"pet_{cell}" for cell in cell_lines] + ["missing_interactions"]
    merged = merged.drop(columns=drop_cols)

    return merged


def sum_rpkm_features_per_cell(
    rpkm_features_dir: str,
    cell_lines: list[str],
    chromatin_marks: list[str]
) -> pd.DataFrame:
    """
    Sum RPKM signal counts for each chromatin feature in each cell line.

    Parameters
    ----------
    rpkm_features_dir : str
        Directory containing RPKM feature data for each cell line.
    cell_lines : list[str]
        List of cell line names.
    chromatin_marks : list[str]
        List of chromatin feature names (e.g., histone marks).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by cell line, with summed feature counts as columns.
    """
    summary_df = pd.DataFrame(columns=chromatin_marks, index=cell_lines)

    for cell in cell_lines:
        for mark in chromatin_marks:
            file_path = f"{rpkm_features_dir}/{cell}/{mark}-W200-G600-FDR1e-05-island.bed"
            reads = pd.read_csv(file_path, sep="\t", names=["chr", "start", "end", "count", "misc"])
            summary_df.at[cell, mark] = reads["count"].sum()

    return summary_df


def normalize_rpkm_features(
    rpkm_intervals_dir: str,
    cell_line: str,
    rpkm_summary: pd.DataFrame,
    chromatin_marks: list[str]
) -> pd.DataFrame:
    """
    Normalize RPKM feature intensities for each interval region in a given cell line.

    Parameters
    ----------
    rpkm_intervals_dir : str
        Directory containing RPKM feature data for chromatin interval regions.
    cell_line : str
        Name of the cell line.
    rpkm_summary : pd.DataFrame
        Summary table of total RPKM counts per feature and cell line.
    chromatin_marks : list[str]
        List of chromatin features.

    Returns
    -------
    pd.DataFrame
        DataFrame of normalized chromatin interval features.
    """
    data = pd.read_csv(f"{rpkm_intervals_dir}/{cell_line}.csv")

    for mark in chromatin_marks:
        data[mark] = (data[mark] / rpkm_summary.at[cell_line, mark]) * 100000

    return data


def compile_interval_chromatin_features(
    rpkm_features_dir: str,
    rpkm_intervals_dir: str,
    cell_lines: list[str],
    chromatin_marks: list[str]
) -> pd.DataFrame:
    """
    Combine normalized chromatin interval features across multiple cell lines.

    Parameters
    ----------
    rpkm_features_dir : str
        Directory containing per-cell RPKM chromatin data.
    rpkm_intervals_dir : str
        Directory containing RPKM interval data.
    cell_lines : list[str]
        List of cell lines.
    chromatin_marks : list[str]
        List of chromatin features.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame of normalized chromatin interval features.
    """
    rpkm_summary = sum_rpkm_features_per_cell(rpkm_features_dir, cell_lines, chromatin_marks)

    all_features = pd.DataFrame()

    for cell in cell_lines:
        normalized = normalize_rpkm_features(rpkm_intervals_dir, cell, rpkm_summary, chromatin_marks)
        normalized["cell_line"] = cell
        all_features = pd.concat([all_features, normalized])

    return all_features.rename(columns={"sequence_name1": "sequence_name"})


def collect_node_chromatin_features(
    motif_features_dir: str,
    cell_lines: list[str]
) -> pd.DataFrame:
    """
    Collect motif-level chromatin features across multiple cell lines.

    Parameters
    ----------
    motif_features_dir : str
        Directory containing motif-level chromatin overlap files.
    cell_lines : list[str]
        List of cell line names.

    Returns
    -------
    pd.DataFrame
        Combined motif chromatin feature DataFrame across all cell lines.
    """
    all_nodes = pd.DataFrame()

    for cell in cell_lines:
        file_path = f"{motif_features_dir}/{cell}__motifs_overlap.csv"
        df = pd.read_csv(file_path)
        df["cell_line"] = cell
        all_nodes = pd.concat([all_nodes, df])

    return all_nodes.fillna(0)


def compile_chromatin_features(
    rpkm_features_dir: str,
    rpkm_intervals_dir: str,
    motif_features_dir: str,
    cell_lines: list[str],
    chromatin_marks: list[str]
) -> pd.DataFrame:
    """
    Integrate chromatin features from interval and motif levels across all cell lines.

    Parameters
    ----------
    rpkm_features_dir : str
        Directory containing per-cell RPKM chromatin feature files.
    rpkm_intervals_dir : str
        Directory containing interval-based RPKM data.
    motif_features_dir : str
        Directory containing motif-based chromatin feature files.
    cell_lines : list[str]
        List of cell line names.
    chromatin_marks : list[str]
        List of chromatin feature names.

    Returns
    -------
    pd.DataFrame
        DataFrame combining interval and motif chromatin features.
    """
    interval_features = compile_interval_chromatin_features(
        rpkm_features_dir, rpkm_intervals_dir, cell_lines, chromatin_marks
    )
    node_features = collect_node_chromatin_features(motif_features_dir, cell_lines)

    merged = interval_features.merge(
        node_features,
        left_on=["sequence_name", "start1", "cell_line"],
        right_on=["sequence_name", "start", "cell_line"],
        suffixes=("_interval", "_node1")
    ).drop(columns=["start"])

    merged = merged.merge(
        node_features,
        left_on=["sequence_name", "stop2", "cell_line"],
        right_on=["sequence_name", "stop", "cell_line"],
        suffixes=("_1", "_2")
    )

    rename_dict = {col: f"{col}_node2" for col in chromatin_marks if col in merged.columns}
    merged = merged.rename(rename_dict, axis=1)

    merged = merged.rename(columns={"start": "start2", "stop_1": "stop1"}).drop(columns=["stop_2"])

    return merged


def compute_conservation_features(
    motif_interaction_dir: str,
    chipseq_motif_dir: str,
    cell_lines: list[str]
) -> pd.DataFrame:
    """
    Compute motif conservation features across cell lines based on motif-interaction overlaps.

    Parameters
    ----------
    motif_interaction_dir : str
        Directory containing intersection files between motifs and ChIA-PET interactions.
    chipseq_motif_dir : str
        Directory containing CTCF ChIP-Seq motif files.
    cell_lines : list[str]
        List of cell line names.

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing motif conservation and degree features.
    """
    first_cell = cell_lines[0]
    df1 = pd.read_csv(f"{motif_interaction_dir}/{first_cell}_rep1_motif_interaction_overlap.csv")
    df2 = pd.read_csv(f"{motif_interaction_dir}/{first_cell}_rep2_motif_interaction_overlap.csv")

    combined = df1.merge(df2, on=["sequence_name", "start", "stop"], suffixes=("1", "2"), how="inner")

    motifs = pd.read_csv(f"{chipseq_motif_dir}/CTCT_motif_{first_cell}.csv")
    combined = motifs[["sequence_name", "start", "stop"]].drop_duplicates().merge(combined)

    for cell in cell_lines[1:]:
        df1 = pd.read_csv(f"{motif_interaction_dir}/{cell}_rep1_motif_interaction_overlap.csv")
        df2 = pd.read_csv(f"{motif_interaction_dir}/{cell}_rep2_motif_interaction_overlap.csv")
        df = df1.merge(df2, on=["sequence_name", "start", "stop"], suffixes=("1", "2"), how="inner")

        motifs = pd.read_csv(f"{chipseq_motif_dir}/CTCT_motif_{cell}.csv")
        df = motifs[["sequence_name", "start", "stop"]].drop_duplicates().merge(df)

        combined = combined.merge(df, on=["sequence_name", "start", "stop"], how="outer")

    degree_cols = [f"degree_{c}" for c in cell_lines]
    nan_cols = [f"{c}_nan" for c in cell_lines]

    combined["all_nan_n"] = combined.isna().sum(axis=1)
    combined = combined.drop_duplicates()

    for cell in cell_lines:
        combined[f"{cell}_nan"] = 2 - combined[[f"start_{cell}1", f"start_{cell}2"]].isna().sum(axis=1)
        combined[f"degree_{cell}"] = combined[[f"degree_{cell}1", f"degree_{cell}2"]].min(axis=1)

    combined = combined.fillna(0)
    combined = combined[["sequence_name", "start", "stop"] + degree_cols + nan_cols].drop_duplicates()

    return combined


def merge_conservation_with_interactions(
    interactions_df: pd.DataFrame,
    motif_interaction_dir: str,
    chipseq_motif_dir: str,
    cell_lines: list[str]
) -> pd.DataFrame:
    """
    Merge conservation features into the motif interaction data.

    Parameters
    ----------
    interactions_df : pd.DataFrame
        DataFrame containing motif pair interactions.
    motif_interaction_dir : str
        Directory containing motif–ChIA-PET intersection files.
    chipseq_motif_dir : str
        Directory containing CTCF ChIP-Seq motif files.
    cell_lines : list[str]
        List of cell line names.

    Returns
    -------
    pd.DataFrame
        DataFrame enriched with motif conservation features.
    """
    conservation_df = compute_conservation_features(motif_interaction_dir, chipseq_motif_dir, cell_lines)

    merged = interactions_df.merge(
        conservation_df,
        left_on=["sequence_name", "start1", "stop1"],
        right_on=["sequence_name", "start", "stop"]
    ).drop(["start", "stop"], axis=1)

    merged = merged.merge(
        conservation_df,
        left_on=["sequence_name", "start2", "stop2"],
        right_on=["sequence_name", "start", "stop"],
        suffixes=("_1", "_2")
    ).drop(["start", "stop"], axis=1)

    return merged
