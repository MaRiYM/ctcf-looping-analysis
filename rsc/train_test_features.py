import pandas as pd

def compute_node_conservation_features(
    pairs_df: pd.DataFrame,
    cell_lines: list[str],
    test_cell_line: str
) -> pd.DataFrame:
    """
    Compute node-level conservation features for motif pair interactions across multiple cell lines.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        DataFrame containing motif pair interaction features across cell lines.
    cell_lines : list of str
        List of all available cell line names.
    test_cell_line : str
        Name of the cell line used as the test set (excluded from training computations).

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with node conservation features added.
    """
    for current_cell in cell_lines:
        other_cells = [c for c in cell_lines if c not in {test_cell_line, current_cell}]

        degree_cols_node1 = [f"degree_{cell}_1" for cell in other_cells]
        nan_cols_node1 = [f"{cell}_nan_1" for cell in other_cells]

        degree_cols_node2 = [f"degree_{cell}_2" for cell in other_cells]
        nan_cols_node2 = [f"{cell}_nan_2" for cell in other_cells]

        mask = pairs_df["cell_line"] == current_cell

        pairs_df.loc[mask, "ave_degree1"] = pairs_df[degree_cols_node1].mean(axis=1)
        pairs_df.loc[mask, "std_degree1"] = pairs_df[degree_cols_node1].std(axis=1)
        pairs_df.loc[mask, "all_nan_n_1"] = pairs_df[nan_cols_node1].sum(axis=1) / 2

        pairs_df.loc[mask, "ave_degree2"] = pairs_df[degree_cols_node2].mean(axis=1)
        pairs_df.loc[mask, "std_degree2"] = pairs_df[degree_cols_node2].std(axis=1)
        pairs_df.loc[mask, "all_nan_n_2"] = pairs_df[nan_cols_node2].sum(axis=1) / 2

    return pairs_df.fillna(0)


def compute_pair_conservation_features(
    pairs_df: pd.DataFrame,
    cell_lines: list[str],
    test_cell_line: str
) -> pd.DataFrame:
    """
    Compute pair-level conservation features such as average and standard deviation of PET values.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        DataFrame containing motif pair interaction features.
    cell_lines : list of str
        List of all available cell line names.
    test_cell_line : str
        Name of the cell line used as the test set.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with pair conservation features added.
    """
    for current_cell in cell_lines:
        other_cells = [c for c in cell_lines if c not in {test_cell_line, current_cell}]
        pet_columns = [f"pet_{cell}" for cell in other_cells]

        mask = pairs_df["cell_line"] == current_cell

        pairs_df.loc[mask, "ave_pet"] = pairs_df[pet_columns].mean(axis=1)
        pairs_df.loc[mask, "std_pet"] = pairs_df[pet_columns].std(axis=1)
        pairs_df.loc[mask, "all_int_cell"] = (
            pairs_df[pet_columns].isin([0]).sum(axis=1) * -1 + len(pet_columns)
        )

    return pairs_df


def compute_conservation_features(
    pairs_df: pd.DataFrame,
    cell_lines: list[str],
    test_cell_line: str
) -> pd.DataFrame:
    """
    Compute both node and pair conservation features for motif pairs.

    Returns
    -------
    pd.DataFrame
        DataFrame with all conservation features filled and updated.
    """
    pairs_df = compute_node_conservation_features(pairs_df, cell_lines, test_cell_line)
    pairs_df = compute_pair_conservation_features(pairs_df, cell_lines, test_cell_line)
    return pairs_df.fillna(0)


def label_interactions(
    data_df: pd.DataFrame,
    cell_lines: list[str],
    chromatin_features: list[str]
) -> pd.DataFrame:
    """
    Label interactions as positive (1) or negative (0) for each cell line.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing PET interaction scores per cell line.
    cell_lines : list of str
        List of cell line names.
    chromatin_features : list of str
        List of chromatin-related feature base names.

    Returns
    -------
    pd.DataFrame
        DataFrame with labeled samples and selected feature columns.
    """
    positive_samples = pd.DataFrame()
    negative_samples = pd.DataFrame()

    for cell_line in cell_lines:
        positive = data_df.query("cell_line == @cell_line and pet_{} > 0".format(cell_line))
        negative = data_df.query("cell_line == @cell_line and pet_{} == 0".format(cell_line))
        positive_samples = pd.concat([positive_samples, positive])
        negative_samples = pd.concat([negative_samples, negative])

    positive_samples["label"] = 1
    negative_samples["label"] = 0

    labeled_data = pd.concat([positive_samples, negative_samples])

    features_int = [f"{f}_interval" for f in chromatin_features] + [
        "all_int_cell", "nodes_pos", "distance", "ave_pet", "std_pet"
    ]
    features_node1 = [f"{f}_node1" for f in chromatin_features] + [
        "score_1", "p-value_1", "q-value_1", "ave_degree1", "std_degree1"
    ]
    features_node2 = [f"{f}_node2" for f in chromatin_features] + [
        "score_2", "p-value_2", "q-value_2", "ave_degree2", "std_degree2"
    ]

    return labeled_data[
        ["sequence_name", "start1", "stop1", "start2", "stop2", "cell_line", "label"]
        + features_int
        + features_node1
        + features_node2
    ]


def train_test_split_by_cell(
    pairs_df: pd.DataFrame,
    cell_lines: list[str],
    test_cell_line: str,
    chromatin_features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and testing sets by cell line.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        The full dataset of motif pair interactions.
    cell_lines : list of str
        List of all cell lines.
    test_cell_line : str
        The cell line used for testing.
    chromatin_features : list of str
        List of chromatin-related features.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Training and testing DataFrames with labeled data.
    """
    data_df = compute_conservation_features(pairs_df, cell_lines, test_cell_line)

    train_df = data_df.query("cell_line != @test_cell_line")
    test_df = data_df.query("cell_line == @test_cell_line")

    train_cell_lines = [c for c in cell_lines if c != test_cell_line]
    test_cell_lines = [test_cell_line]

    train_labeled = label_interactions(train_df, train_cell_lines, chromatin_features)
    test_labeled = label_interactions(test_df, test_cell_lines, chromatin_features)

    return train_labeled, test_labeled
