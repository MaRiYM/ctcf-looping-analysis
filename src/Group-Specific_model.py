import argparse
import warnings
warnings.filterwarnings("ignore")
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
from neighboring_features import data_neighboring_features


def prepare_all_features(
    chia_pet_dir: str,
    chipseq_dir: str,
    chromatin_features_dir: str,
    cell_lines: list[str],
    chromatin_features: list[str],
    path_rpkm_features: str,
    path_rpkm_intervals: str,
    path_intersect_motif_chia: str,
    path_ctcf_motifs: str,
) -> pd.DataFrame:
    """Prepare the full dataset combining interaction, chromatin, and conservation features."""
    cell_lines_copy = list(cell_lines)

    print("Step 1: Preparing motif interaction data...")
    all_cells_interactions = prepare_interaction_data(chia_pet_dir, chipseq_dir, cell_lines_copy)

    print("Step 2: Adding interaction distance and strand features...")
    interaction_features_df = add_interaction_distance_and_strand_features(
        all_cells_interactions, chromatin_features_dir, cell_lines_copy
    )

    print("Step 3: Compiling chromatin features...")
    chromatin_features_df = compile_chromatin_features(
        path_rpkm_features,
        path_rpkm_intervals,
        chromatin_features_dir,
        cell_lines_copy,
        chromatin_features,
    )

    print("Step 4: Merging conservation features...")
    conservation_features_df = merge_conservation_with_interactions(
        all_cells_interactions,
        path_intersect_motif_chia,
        path_ctcf_motifs,
        cell_lines_copy,
    )

    print("Step 5: Combining all features into one dataset...")
    all_features_df = (
        conservation_features_df
        .merge(interaction_features_df, on=["sequence_name", "start1", "stop1", "start2", "stop2"])
        .merge(chromatin_features_df, on=["sequence_name", "start1", "stop1", "start2", "stop2"])
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run motif interaction prediction pipeline")

    parser.add_argument("--motif_chromatin", required=True, help="Path to motif chromatin features directory")
    parser.add_argument("--intersect_chiapet", required=True, help="Path to intersections between motifs and ChIA-PET data")
    parser.add_argument("--intersect_chipseq", required=True, help="Path to intersections between motifs and ChIP-Seq data")
    parser.add_argument("--rpkm_features", required=True, help="Path to RPKM features directory")
    parser.add_argument("--rpkm_intervals", required=True, help="Path to RPKM intervals directory")
    parser.add_argument("--intersect_motif_chia", required=True, help="Path to motif–ChIA-PET overlaps")
    parser.add_argument("--ctcf_motifs", required=True, help="Path to CTCF motifs directory")
    parser.add_argument("--test_cell", required=True, help="Cell line to use as test")
    parser.add_argument("--features_group", type=int, help="Neighboring feature group size")
    parser.add_argument("--cell_lines", nargs="+", required=True, help="List of cell lines")

    args = parser.parse_args()

    chromatin_features = [
        "H3K4me1", "H3K4me2", "H3K4me3", "H3K9me3", "RAD21", "CTCF",
        "H3K36me3", "H3K79me2", "H3K27ac", "H3K9ac", "H3K27me3", "H2AFZ", "H4K20me1"
    ]

    cell_lines = args.cell_lines

    pairs_df = prepare_all_features(
        args.intersect_chiapet,
        args.intersect_chipseq,
        args.motif_chromatin,
        cell_lines,
        chromatin_features,
        args.rpkm_features,
        args.rpkm_intervals,
        args.intersect_motif_chia,
        args.ctcf_motifs,
    )

    train_df, test_df = train_test_split_by_cell(
        pairs_df,
        cell_lines,
        args.test_cell,
        chromatin_features,
    )

    print(f"Computing neighboring features (group={args.features_group})...")
    train_neighboring_df = data_neighboring_features(train_df, args.features_group)
    test_neighboring_df = data_neighboring_features(test_df, args.features_group)

    test_confidence_df = train_and_evaluate_model(train_neighboring_df, test_neighboring_df)

    print("✅ Pipeline complete.")
