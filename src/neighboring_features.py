import pandas as pd

# ============================================================
# ======== 1. INTERACTION INFO FOR COMPUTED DATA ==============
# ============================================================

def interaction_info(group_cols: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute interaction statistics (positive, negative, average probabilities)
    for predicted motif interactions (computed data).
    """
    pos_int = df.groupby(group_cols, as_index=False)["prediction"].sum()
    all_int = df.groupby(group_cols, as_index=False).size()
    ints = all_int.merge(pos_int, on=group_cols)
    ints["neg_int"] = ints["size"] - ints["prediction"]

    ave_int = df.groupby(group_cols + ["prediction"], as_index=False)["probability"].mean()
    ave_ints = (
        ave_int.pivot(index=group_cols, columns="prediction", values="probability")
        .reset_index()
        .rename(columns={0: "neg_prob", 1: "pos_prob"})
    )

    ave_info = ave_ints.merge(ints, on=group_cols, how="outer").fillna(0)
    return ave_info


# ============================================================
# ======== 2. INTERACTION INFO FOR DATA-DRIVEN DATA ===========
# ============================================================

def interaction_info2(group_cols: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute interaction statistics (positive, negative, total)
    for experimental/data-driven interactions (using labels).
    """
    df = df.copy()
    pos_int = df.groupby(group_cols, as_index=False)['label'].sum().rename(columns={'label': 'label'})
    all_int = df.groupby(group_cols, as_index=False).size().rename(columns={'size': 'total_int'})
    ints = all_int.merge(pos_int, on=group_cols, how='outer').fillna(0)
    ints['neg_int'] = ints['total_int'] - ints['label']
    return ints.fillna(0)


# ============================================================
# ======== 3. COMMON FEATURE GENERATION ======================
# ============================================================

def find_features(int_info_df: pd.DataFrame, all_nodes: pd.DataFrame, feature_cols: tuple) -> pd.DataFrame:
    """
    Generate neighboring node-based features (shifts + distances).
    Works for both computed and data-driven interaction data.
    """
    nodes_info = all_nodes.merge(int_info_df, how="outer").fillna(0)
    nodes_info = (
        nodes_info.sort_values(["sequence_name", "start1", "stop1"])
        .reset_index(drop=True)[
            ["sequence_name", "start1", "stop1"] + list(feature_cols)
        ]
    )

    shifts = [1, 2, 3, 4, 5]

    # Helper to add shifted features
    def add_shifted(df, col, prefix, shifts):
        for i in shifts:
            df[f"{prefix}{i}"] = df.groupby("sequence_name")[col].shift(i)
        return df

    # --- LEFT (upstream/downstream) ---
    for col, prefix in zip(
        feature_cols,
        [
            "dergee_pos_upstream_left",
            "dergee_neg_upstream_left",
            "dergee_pos_downstream_left",
            "dergee_neg_downstream_left",
        ],
    ):
        add_shifted(nodes_info, col, prefix, shifts)

    # Distances left
    for i in shifts:
        nodes_info[f"s{i}"] = nodes_info.groupby("sequence_name")["start1"].shift(i)
    nodes_info = nodes_info.fillna(-1)
    for i in shifts:
        nodes_info[f"distance_left{i}"] = (nodes_info["start1"] - nodes_info[f"s{i}"]) * -1
    nodes_info = nodes_info.drop(columns=[f"s{i}" for i in shifts])

    # --- RIGHT (upstream/downstream) ---
    for col, prefix in zip(
        feature_cols,
        [
            "dergee_pos_upstream_right",
            "dergee_neg_upstream_right",
            "dergee_pos_downstream_right",
            "dergee_neg_downstream_right",
        ],
    ):
        add_shifted(nodes_info, col, prefix, [-i for i in shifts])

    for i in shifts:
        nodes_info[f"s{i}"] = nodes_info.groupby("sequence_name")["start1"].shift(-i)
    nodes_info = nodes_info.fillna(-1)
    for i in shifts:
        nodes_info[f"distance_right{i}"] = nodes_info[f"s{i}"] - nodes_info["start1"]
    nodes_info = nodes_info.drop(columns=[f"s{i}" for i in shifts])

    return nodes_info


# ============================================================
# ======== 4. COMPUTED NEIGHBORING FEATURES ==================
# ============================================================

def computed_neighboring_features(X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Generate neighboring interaction features for computed (predicted) data.
    """
    neighboring_group_loops = X_test.query("probability > 0.95 or probability < 0.05")

    # Extract unique nodes
    node_pair1 = X_test[["sequence_name", "start1", "stop1"]].drop_duplicates()
    node_pair2 = X_test[["sequence_name", "start2", "stop2"]].drop_duplicates().rename(
        columns={"start2": "start1", "stop2": "stop1"}
    )
    all_test_nodes = pd.concat([node_pair1, node_pair2]).drop_duplicates()

    # Compute left/right info
    node_pair1_info = interaction_info(["sequence_name", "start1", "stop1"], neighboring_group_loops)
    node_pair2_info = interaction_info(["sequence_name", "start2", "stop2"], neighboring_group_loops)
    node_pair2_info = node_pair2_info.rename(columns={"start2": "start1", "stop2": "stop1"})

    # Merge and adjust
    all_nodes_info = node_pair1_info.merge(
        node_pair2_info, on=["sequence_name", "start1", "stop1"], how="outer", suffixes=["_L", "_R"]
    ).drop(columns=["size_L", "size_R"], errors="ignore")

    all_nodes_info.loc[all_nodes_info["neg_int_L"] != 0, "neg_prob_L"] = 100 - all_nodes_info["neg_prob_L"]
    all_nodes_info.loc[all_nodes_info["neg_int_R"] != 0, "neg_prob_R"] = 100 - all_nodes_info["neg_prob_R"]

    nodes_interaction_info_df = all_nodes_info.fillna(0)

    # Features — columns for computed version
    feature_cols = ("prediction_L", "neg_int_L", "prediction_R", "neg_int_R")

    neighboring_features = find_features(nodes_interaction_info_df, all_test_nodes, feature_cols)
    neighboring_features = neighboring_features.rename(columns={"start1": "start", "stop1": "stop"})

    # Merge with test set
    merged_1 = X_test.merge(
        neighboring_features,
        left_on=["sequence_name", "start1", "stop1"],
        right_on=["sequence_name", "start", "stop"],
    ).drop(columns=["start", "stop"])

    merged_2 = merged_1.merge(
        neighboring_features,
        left_on=["sequence_name", "start2", "stop2"],
        right_on=["sequence_name", "start", "stop"],
        suffixes=["_1", "_2"],
    ).drop(columns=["start", "stop"])

    return merged_2.fillna(0)


# ============================================================
# ======== 5. DATA-DRIVEN NEIGHBORING FEATURES ===============
# ============================================================

def data_neighboring_features(X_test: pd.DataFrame, neighboring_features_group: int) -> pd.DataFrame:
    """
    Generate neighboring interaction features for data-driven (experimental) data.
    """
    neighboring_group_loops = X_test.query('all_int_cell == @neighboring_features_group')

    # Extract unique nodes
    node_pair1 = X_test[["sequence_name", "start1", "stop1"]].drop_duplicates()
    node_pair2 = X_test[["sequence_name", "start2", "stop2"]].drop_duplicates().rename(
        columns={"start2": "start1", "stop2": "stop1"}
    )
    all_test_nodes = pd.concat([node_pair1, node_pair2]).drop_duplicates()

    # Compute left/right info
    node_pair1_info = interaction_info2(["sequence_name", "start1", "stop1"], neighboring_group_loops)
    node_pair2_info = interaction_info2(["sequence_name", "start2", "stop2"], neighboring_group_loops)
    node_pair2_info = node_pair2_info.rename(columns={"start2": "start1", "stop2": "stop1"})

    all_nodes_info = node_pair1_info.merge(
        node_pair2_info,
        on=["sequence_name", "start1", "stop1"],
        how="outer",
        suffixes=["_L", "_R"],
    ).fillna(0)

    # Features — columns for data-driven version
    feature_cols = ("label_L", "neg_int_L", "label_R", "neg_int_R")

    neighboring_features = find_features(nodes_info_df := all_nodes_info, all_test_nodes, feature_cols)
    neighboring_features = neighboring_features.rename(columns={"start1": "start", "stop1": "stop"})

    # Merge with test data
    merged_1 = X_test.merge(
        neighboring_features,
        left_on=["sequence_name", "start1", "stop1"],
        right_on=["sequence_name", "start", "stop"],
    ).drop(columns=["start", "stop"])

    merged_2 = merged_1.merge(
        neighboring_features,
        left_on=["sequence_name", "start2", "stop2"],
        right_on=["sequence_name", "start", "stop"],
        suffixes=["_1", "_2"],
    ).drop(columns=["start", "stop"])

    return merged_2.fillna(0)
