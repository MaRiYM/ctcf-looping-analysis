#!/usr/bin/env python3
"""
Two‑step chromatin interaction prediction pipeline.

This script first predicts interactions using a Gradient Boosting model,
then extracts neighboring features from the predictions and retrains a model
on those features for a final evaluation.

Usage:
    python two_step_pipeline.py \
        --chia-pet-dir /path/to/intersect_chiapet \
        --chipseq-dir /path/to/intersect_chipseq \
        --chromatin-dir /path/to/motif_chromatin_features \
        --rpkm-features-dir /path/to/rpkm_features \
        --rpkm-intervals-dir /path/to/rpkm_intervals \
        --intersect-motif-chia-dir /path/to/intersect_motif_chia \
        --ctcf-motifs-dir /path/to/ctcf_motifs \
        --test-cell-line GM \
        --cell-lines GM H1 HCT116 HepG2 IMR90 K562 MCF7 SKNSH \
        --chromatin-features H3K4me1 H3K4me2 H3K4me3 H3K9me3 RAD21 CTCF \
                            H3K36me3 H3K79me2 H3K27ac H3K9ac H3K27me3 H2AFZ H4K20me1 \
        --output-dir ./outputs
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Custom modules (adjust import paths if needed)
from data_preparation import prepare_interaction_data
from train_test_features import train_test_split_by_cell
from features_enginearing import (
    add_interaction_distance_and_strand_features,
    compile_chromatin_features,
    merge_conservation_with_interactions,
)
from neighboring_features import computed_neighboring_features, data_neighboring_features

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def fast_overlaps_with_percentage(df1, df2):
    """
    Fast overlap detection using sorting and sweep line algorithm.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames with columns 'sequence_name', 'start1', 'stop2'

    Returns
    -------
    pd.DataFrame
        Overlapping pairs with overlap percentages.
    """
    results = []

    for chrom in df1['sequence_name'].unique():
        intervals1 = [(row['start1'], row['stop2'], idx, row)
                      for idx, row in df1[df1['sequence_name'] == chrom].iterrows()]
        intervals2 = [(row['start1'], row['stop2'], idx, row)
                      for idx, row in df2[df2['sequence_name'] == chrom].iterrows()]

        if not intervals2:
            continue

        intervals1.sort(key=lambda x: x[0])
        intervals2.sort(key=lambda x: x[0])

        active_intervals = []
        i = j = 0

        while i < len(intervals1) and j < len(intervals2):
            start1, end1, idx1, row1 = intervals1[i]
            start2, end2, idx2, row2 = intervals2[j]

            if start1 < start2:
                for act_start, act_end, act_idx, act_row in active_intervals:
                    overlap_start = max(start1, act_start)
                    overlap_end = min(end1, act_end)
                    if overlap_start < overlap_end:
                        _add_overlap_result(results, chrom, row1, act_row,
                                            overlap_start, overlap_end)
                i += 1
            else:
                active_intervals.append((start2, end2, idx2, row2))
                active_intervals = [(s, e, idx, row) for s, e, idx, row in active_intervals
                                    if e > start1]
                j += 1

        while i < len(intervals1):
            start1, end1, idx1, row1 = intervals1[i]
            for act_start, act_end, act_idx, act_row in active_intervals:
                overlap_start = max(start1, act_start)
                overlap_end = min(end1, act_end)
                if overlap_start < overlap_end:
                    _add_overlap_result(results, chrom, row1, act_row,
                                        overlap_start, overlap_end)
            i += 1

    return pd.DataFrame(results)


def _add_overlap_result(results, chrom, row1, row2, overlap_start, overlap_end):
    """Helper to add overlap result with percentages."""
    overlap_len = overlap_end - overlap_start
    len1 = row1['stop2'] - row1['start1']
    len2 = row2['stop2'] - row2['start1']

    result = {
        'sequence_name': chrom,
        'df1_start': row1['start1'],
        'df1_stop': row1['stop2'],
        'df2_start': row2['start1'],
        'df2_stop': row2['stop2'],
        'overlap_length': overlap_len,
        'percent_of_df1': (overlap_len / len1) * 100 if len1 > 0 else 0,
        'percent_of_df2': (overlap_len / len2) * 100 if len2 > 0 else 0,
    }

    for col in row1.index:
        if col not in ['sequence_name', 'start1', 'stop2']:
            result[f'df1_{col}'] = row1[col]
    for col in row2.index:
        if col not in ['sequence_name', 'start1', 'stop2']:
            result[f'df2_{col}'] = row2[col]

    results.append(result)


# ----------------------------------------------------------------------
# Core pipeline functions
# ----------------------------------------------------------------------

def prepare_all_features(
    chia_pet_dir: str,
    chipseq_dir: str,
    chromatin_dir: str,
    rpkm_features_dir: str,
    rpkm_intervals_dir: str,
    intersect_motif_chia_dir: str,
    ctcf_motifs_dir: str,
    cell_lines: list,
    chromatin_features: list,
) -> pd.DataFrame:
    """Prepare the full dataset combining interaction, chromatin, and conservation features."""
    logging.info("Step 1: Preparing motif interaction data...")
    all_cells_interactions = prepare_interaction_data(
        chia_pet_dir, chipseq_dir, cell_lines
    )

    logging.info("Step 2: Adding interaction distance and strand features...")
    interaction_features_df = add_interaction_distance_and_strand_features(
        all_cells_interactions,
        chromatin_dir,
        cell_lines,
    )

    logging.info("Step 3: Compiling chromatin features...")
    chromatin_features_df = compile_chromatin_features(
        rpkm_features_dir,
        rpkm_intervals_dir,
        chromatin_dir,
        cell_lines,
        chromatin_features,
    )

    logging.info("Step 4: Merging conservation features...")
    conservation_features_df = merge_conservation_with_interactions(
        all_cells_interactions,
        intersect_motif_chia_dir,
        ctcf_motifs_dir,
        cell_lines,
    )

    logging.info("Step 5: Combining all features into one dataset...")
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

    logging.info("✅ Feature preparation complete.")
    return all_features_df


def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test,
                             test_df, model_name, output_dir) -> pd.DataFrame:
    """
    Train a Gradient Boosting model and evaluate performance.
    """
    logging.info("Training Gradient Boosting Classifier...")

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        min_samples_split=30,
        min_samples_leaf=15,
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]

    logging.info("\n" + "=" * 60)
    logging.info("MODEL PERFORMANCE")
    logging.info("=" * 60)
    logging.info(f"Train Accuracy: {accuracy_score(y_train, train_pred):.4f}")
    logging.info(f"Validation Accuracy: {accuracy_score(y_val, val_pred):.4f}")
    logging.info(f"Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")

    logging.info("\n" + "=" * 60)
    logging.info("CLASSIFICATION REPORT - TEST SET")
    logging.info("=" * 60)
    logging.info("\n" + classification_report(y_test, test_pred))

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    imp_path = os.path.join(output_dir, f'feature_importance_{model_name}.csv')
    feature_importance.to_csv(imp_path, index=False)
    logging.info(f"Feature importance saved to {imp_path}")

    # Top features
    logging.info("\n" + "=" * 60)
    logging.info("TOP 15 MOST IMPORTANT FEATURES")
    logging.info("=" * 60)
    for idx, row in feature_importance.head(15).iterrows():
        logging.info(f"{idx+1:2d}. {row['feature']:40s} {row['importance']:.6f}")

    # Add predictions to test dataframe
    test_df = test_df.copy()
    test_df["probability"] = test_proba
    test_df["prediction"] = test_pred
    return test_df


def apply_group_aware_sampling(X, y, groups):
    """
    Apply SMOTE to groups that are imbalanced, while preserving group structure.
    """
    X_balanced_list = []
    y_balanced_list = []
    groups_balanced_list = []

    for group in sorted(groups.unique()):
        mask = groups == group
        X_group = X[mask]
        y_group = y[mask]

        n_pos = (y_group == 1).sum()
        n_neg = (y_group == 0).sum()
        ratio = n_neg / n_pos if n_pos > 0 else float('inf')

        if ratio > 2 and n_pos > 1:
            try:
                smote = SMOTE(sampling_strategy='auto', random_state=42,
                              k_neighbors=min(3, n_pos - 1))
                X_bal, y_bal = smote.fit_resample(X_group, y_group)
                logging.info(f"  Group {group}: SMOTE applied (ratio {ratio:.1f}:1)")
            except Exception as e:
                logging.warning(f"  Group {group}: SMOTE failed ({e}), using original")
                X_bal, y_bal = X_group, y_group
        else:
            X_bal, y_bal = X_group, y_group
            logging.info(f"  Group {group}: No sampling needed (ratio {ratio:.1f}:1)")

        X_balanced_list.append(X_bal)
        y_balanced_list.append(y_bal)
        groups_balanced_list.append(pd.Series([group] * len(y_bal), index=y_bal.index))

    X_balanced = pd.concat(X_balanced_list, axis=0)
    y_balanced = pd.concat(y_balanced_list, axis=0)
    groups_balanced = pd.concat(groups_balanced_list, axis=0)

    return X_balanced, y_balanced, groups_balanced


def prepare_training_data(train_df, non_predictors):
    """Split and apply group-aware sampling to training data."""
    group_column = train_df['all_int_cell'].copy()
    X = train_df.drop(columns=non_predictors)
    y = train_df["label"]

    X_train, X_val, y_train, y_val, group_train, group_val = train_test_split(
        X, y, group_column,
        test_size=0.2,
        random_state=42,
        stratify=group_column,
    )

    logging.info("\n" + "=" * 60)
    logging.info("APPLYING SMART SAMPLING (Preserving Group Features)")
    logging.info("=" * 60)

    X_train_bal, y_train_bal, groups_bal = apply_group_aware_sampling(
        X_train, y_train, group_train
    )

    logging.info(f"Before sampling: {X_train.shape}, Pos={(y_train==1).sum()}, Neg={(y_train==0).sum()}")
    logging.info(f"After sampling: {X_train_bal.shape}, Pos={(y_train_bal==1).sum()}, Neg={(y_train_bal==0).sum()}")

    return X_train_bal, y_train_bal, X_val, y_val


def run_prediction_pipeline(
    paths,
    cell_lines,
    chromatin_features,
    test_cells,
    non_predictors,
    output_dir,
    step_name="step1",
):
    """
    Run the first prediction step for a list of test cell lines.
    Returns a concatenated DataFrame of neighboring features for all processed cells.
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"RUNNING {step_name.upper()} PREDICTION FOR CELLS: {test_cells}")
    logging.info(f"{'='*60}")

    # Prepare full feature set once
    pairs_df = prepare_all_features(
        paths['chia_pet_dir'],
        paths['chipseq_dir'],
        paths['chromatin_dir'],
        paths['rpkm_features_dir'],
        paths['rpkm_intervals_dir'],
        paths['intersect_motif_chia_dir'],
        paths['ctcf_motifs_dir'],
        cell_lines,
        chromatin_features,
    )

    all_neighboring = pd.DataFrame()

    for test_cell in test_cells:
        logging.info(f"\n--- Processing test cell: {test_cell} ---")

        # Split by cell
        train_df, test_df = train_test_split_by_cell(
            pairs_df, cell_lines, test_cell, chromatin_features
        )

        # Keep only groups with all_int_cell > 0 (optional)
        test_df = test_df.query('all_int_cell != 0')

        # Prepare training data with sampling
        X_train, y_train, X_val, y_val = prepare_training_data(train_df, non_predictors)

        # Test data
        X_test = test_df.drop(columns=non_predictors)
        y_test = test_df["label"]

        # Train and evaluate
        test_df = train_and_evaluate_model(
            X_train, y_train, X_val, y_val, X_test, y_test,
            test_df, f"{step_name}_{test_cell}", output_dir
        )

        # Save predictions
        test_df.to_csv(os.path.join(output_dir, f'{step_name}_{test_cell}.csv'), index=False)

        # Compute neighboring features for this test cell
        neigh_df = computed_neighboring_features(test_df)

        # Overlap with top predictions (example from original code)
        df1 = test_df[['sequence_name', 'start1', 'stop2']]
        df2 = test_df.sort_values('probability', ascending=False)[:2000][['sequence_name', 'start1', 'stop2']]

        overlap_pairs = fast_overlaps_with_percentage(df1, df2)
        overlap_pairs['max_overlap'] = overlap_pairs[['percent_of_df1', 'percent_of_df2']].max(axis=1)
        overlap_pairs = overlap_pairs.query('percent_of_df1!=100 or percent_of_df2!=100')

        in_df = overlap_pairs.query('percent_of_df1>95').groupby(
            ['sequence_name', 'df1_start', 'df1_stop']
        ).size().to_frame('inerr_loop').reset_index()
        out_df = overlap_pairs.query('percent_of_df2>95').groupby(
            ['sequence_name', 'df1_start', 'df1_stop']
        ).size().to_frame('outer_loop').reset_index()
        cross_df = overlap_pairs.query('max_overlap<95').groupby(
            ['sequence_name', 'df1_start', 'df1_stop']
        ).size().to_frame('cross').reset_index()

        df_in_out = in_df.merge(out_df, how='outer').fillna(0)
        df_cross_int = df_in_out.merge(cross_df, how='outer').fillna(0)
        df_cross_int = df_cross_int.rename({'df1_start': 'start1', 'df1_stop': 'stop2'}, axis=1)

        neigh_df = neigh_df.merge(df_cross_int, how='left').fillna(0)

        all_neighboring = pd.concat([all_neighboring, neigh_df], ignore_index=True)

    return all_neighboring


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two‑step chromatin interaction prediction pipeline."
    )
    parser.add_argument('--chia-pet-dir', required=True,
                        help="Directory with intersect_chiapet folder")
    parser.add_argument('--chipseq-dir', required=True,
                        help="Directory with intersect_chipseqk folder")
    parser.add_argument('--chromatin-dir', required=True,
                        help="Directory with motif_chromatin_features folder")
    parser.add_argument('--rpkm-features-dir', required=True,
                        help="Directory with RPKM features folder")
    parser.add_argument('--rpkm-intervals-dir', required=True,
                        help="Directory with RPKM intervals folder")
    parser.add_argument('--intersect-motif-chia-dir', required=True,
                        help="Directory with intersect_motif_chia folder")
    parser.add_argument('--ctcf-motifs-dir', required=True,
                        help="Directory with CTCF motif folder")
    parser.add_argument('--test-cell-line', default='GM',
                        help="Cell line to use as test (default: GM)")
    parser.add_argument('--cell-lines', nargs='+',
                        default=['GM', 'H1', 'HCT116', 'HepG2', 'IMR90', 'K562', 'MCF7', 'SKNSH'],
                        help="List of cell lines")
    parser.add_argument('--chromatin-features', nargs='+',
                        default=["H3K4me1", "H3K4me2", "H3K4me3", "H3K9me3", "RAD21", "CTCF",
                                 "H3K36me3", "H3K79me2", "H3K27ac", "H3K9ac", "H3K27me3",
                                 "H2AFZ", "H4K20me1"],
                        help="List of chromatin features")
    parser.add_argument('--output-dir', default='./outputs',
                        help="Base output directory (default: ./outputs)")
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    os.makedirs(args.output_dir, exist_ok=True)

    paths = {
        'chia_pet_dir': args.chia_pet_dir,
        'chipseq_dir': args.chipseq_dir,
        'chromatin_dir': args.chromatin_dir,
        'rpkm_features_dir': args.rpkm_features_dir,
        'rpkm_intervals_dir': args.rpkm_intervals_dir,
        'intersect_motif_chia_dir': args.intersect_motif_chia_dir,
        'ctcf_motifs_dir': args.ctcf_motifs_dir,
    }

    NON_PREDICTORS = [
        "sequence_name", "start1", "stop1", "start2", "stop2",
        "label", "cell_line", "all_int_cell"
    ]

    # Step 1: Predict on test cell line
    test_neigh = run_prediction_pipeline(
        paths,
        args.cell_lines,
        args.chromatin_features,
        [args.test_cell_line],
        NON_PREDICTORS,
        args.output_dir,
        step_name="step1",
    )

    # Step 2: Predict on all other cell lines to get training neighboring features
    remaining_cells = [c for c in args.cell_lines if c != args.test_cell_line]
    train_neigh = run_prediction_pipeline(
        paths,
        args.cell_lines,
        args.chromatin_features,
        remaining_cells,
        NON_PREDICTORS,
        args.output_dir,
        step_name="step1",
    )

    # Step 3: Final model trained on neighboring features
    logging.info("\n" + "=" * 60)
    logging.info("FINAL STAGE: Training on neighboring features")
    logging.info("=" * 60)

    X_train, y_train, X_val, y_val = prepare_training_data(train_neigh, NON_PREDICTORS)
    X_test = test_neigh.drop(columns=NON_PREDICTORS)
    y_test = test_neigh["label"]

    final_test_df = train_and_evaluate_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        test_neigh, f"2step_{args.test_cell_line}", args.output_dir
    )
    final_test_df.to_csv(os.path.join(args.output_dir, f'final_2step_{args.test_cell_line}.csv'), index=False)

    logging.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
