import argparse
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Import your custom modules
from data_preparation import prepare_interaction_data
from train_test_features import train_test_split_by_cell
from features_enginearing import (
    add_interaction_distance_and_strand_features,
    compile_chromatin_features,
    merge_conservation_with_interactions,
)
from neighboring_features import computed_neighboring_features, data_neighboring_features


def prepare_all_features(
    chia_pet_dir: str,
    chipseq_dir: str,
    chromatin_features_dir: str,
    chromatin_features: list[str],
    rpkm_features_path: str,
    rpkm_intervals_path: str,
    intersect_motif_chia: str,
    ctcf_motifs_path: str,
    cell_lines: list[str],
) -> pd.DataFrame:
    """Prepare the full dataset combining interaction, chromatin, and conservation features."""

    print("Step 1: Preparing motif interaction data...")
    all_cells_interactions = prepare_interaction_data(chia_pet_dir, chipseq_dir, cell_lines)

    print("Step 2: Adding interaction distance and strand features...")
    interaction_features_df = add_interaction_distance_and_strand_features(
        all_cells_interactions, chromatin_features_dir, cell_lines
    )

    print("Step 3: Compiling chromatin features...")
    chromatin_features_df = compile_chromatin_features(
        rpkm_features_path,
        rpkm_intervals_path,
        chromatin_features_dir,
        cell_lines,
        chromatin_features,
    )

    print("Step 4: Merging conservation features...")
    conservation_features_df = merge_conservation_with_interactions(
        all_cells_interactions,
        intersect_motif_chia,
        ctcf_motifs_path,
        cell_lines,
    )

    print("Step 5: Combining all features into one dataset...")
    all_features_df = (
        conservation_features_df
        .merge(
            interaction_features_df,
            on=["sequence_name", "start1", "stop1", "start2", "stop2"],
        )
        .merge(
            chromatin_features_df,
            on=["sequence_name", "start1", "stop1", "start2", "stop2"],
        )
    )

    print("✅ Feature preparation complete.")
    return all_features_df


def train_and_evaluate_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Train a Gradient Boosting model and evaluate performance on the test set."""

    print("Training Gradient Boosting Classifier...")

    nonpredictors = ["sequence_name", "start1", "stop1", "start2", "stop2", "label", "cell_line"]

    model = GradientBoostingClassifier(
        n_estimators=4000,
        learning_rate=0.1,
        max_depth=5,
        random_state=0,
    )

    X_train = train_df.drop(columns=nonpredictors)
    y_train = train_df["label"]

    X_test = test_df.drop(columns=nonpredictors)
    y_test = test_df["label"]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    print(f"✅ Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()

    test_df["probability"] = probabilities
    test_df["prediction"] = predictions

    return test_df


def step1_prediction(
    PATH_INTERSECT_CHIAPET,
    PATH_INTERSECT_CHIPSEQ,
    PATH_MOTIF_CHROMATIN_FEATURES,
    CELL_LINES,
    CHROMATIN_FEATURES,
    TEST_CELL_LINES,
    PATH_RPKM_FEATURES,
    PATH_RPKM_INTERVALS,
    PATH_INTERSECT_MOTIF_CHIA,
    PATH_CTCF_MOTIFS,
):
    """Perform Step 1 prediction for one or multiple test cell lines."""
    pairs_df = prepare_all_features(
        PATH_INTERSECT_CHIAPET,
        PATH_INTERSECT_CHIPSEQ,
        PATH_MOTIF_CHROMATIN_FEATURES,
        CHROMATIN_FEATURES,
        PATH_RPKM_FEATURES,
        PATH_RPKM_INTERVALS,
        PATH_INTERSECT_MOTIF_CHIA,
        PATH_CTCF_MOTIFS,
        CELL_LINES,
    )

    step2_features_df = pd.DataFrame()
    for TEST_CELL in TEST_CELL_LINES:
        train_df, test_df = train_test_split_by_cell(
            pairs_df, CELL_LINES, TEST_CELL, CHROMATIN_FEATURES
        )
        test_df = train_and_evaluate_model(train_df, test_df)
        test_neighboring_df = computed_neighboring_features(test_df)
        step2_features_df = pd.concat([step2_features_df, test_neighboring_df])

    return step2_features_df


def main():
    parser = argparse.ArgumentParser(description="Run CTCF interaction prediction pipeline.")
    parser.add_argument("--motif_chromatin", required=True, help="Path to Motif Chromatin Features")
    parser.add_argument("--intersect_chiapet", required=True, help="Path to ChIA-PET intersections")
    parser.add_argument("--intersect_chipseq", required=True, help="Path to ChIP-Seq intersections")
    parser.add_argument("--rpkm_features", required=True, help="Path to RPKM features")
    parser.add_argument("--rpkm_intervals", required=True, help="Path to RPKM intervals")
    parser.add_argument("--intersect_motif_chia", required=True, help="Path to Motif-ChIA overlaps")
    parser.add_argument("--ctcf_motifs", required=True, help="Path to CTCF motifs")
    parser.add_argument("--test_cell", required=True, help="Test cell line (e.g., HCT116)")
    parser.add_argument("--cell_lines", nargs="+", required=True, help="List of cell lines")

    args = parser.parse_args()

    CHROMATIN_FEATURES = [
        "H3K4me1", "H3K4me2", "H3K4me3", "H3K9me3", "RAD21", "CTCF",
        "H3K36me3", "H3K79me2", "H3K27ac", "H3K9ac", "H3K27me3", "H2AFZ", "H4K20me1"
    ]

    CELL_LINES = args.cell_lines
    TEST_CELL_LINE = args.test_cell

    print("=== Running Step 1 Prediction for Test Cell Line ===")
    test_neighboring_features = step1_prediction(
        args.intersect_chiapet,
        args.intersect_chipseq,
        args.motif_chromatin,
        CELL_LINES,
        CHROMATIN_FEATURES,
        [TEST_CELL_LINE],
        args.rpkm_features,
        args.rpkm_intervals,
        args.intersect_motif_chia,
        args.ctcf_motifs,
    )

    CELL_LINES.remove(TEST_CELL_LINE)

    print("=== Running Step 1 Prediction for Remaining Cell Lines ===")
    train_neighboring_features = step1_prediction(
        args.intersect_chiapet,
        args.intersect_chipseq,
        args.motif_chromatin,
        CELL_LINES,
        CHROMATIN_FEATURES,
        CELL_LINES,
        args.rpkm_features,
        args.rpkm_intervals,
        args.intersect_motif_chia,
        args.ctcf_motifs,
    )

    print("=== Final Model Training and Evaluation ===")
    test_confidence_df = train_and_evaluate_model(train_neighboring_features, test_neighboring_features)

    test_confidence_df.to_csv("test_confidence_results.csv", index=False)
    print("✅ Results saved to test_confidence_results.csv")


if __name__ == "__main__":
    main()
