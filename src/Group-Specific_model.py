#!/usr/bin/env python3
"""
Group‑specific chromatin interaction modeling pipeline.

This script trains Gradient Boosting classifiers for different feature groups
(defined by the number of interacting cell lines) and evaluates performance.
It uses stratified sampling to balance positive/negative examples within each group,
and optionally runs two versions: one with all features and one with selected
neighbouring features removed.

Usage:
    python run_pipeline.py \
        --chia-pet-dir /path/to/chiapet \
        --chipseq-dir /path/to/chipseq \
        --chromatin-dir /path/to/chromatin \
        --rpkm-features-dir /path/to/rpkm_features \
        --rpkm-intervals-dir /path/to/rpkm_intervals \
        --intersect-motif-chia-dir /path/to/intersect_motif_chia \
        --ctcf-motifs-dir /path/to/ctcf_motifs \
        --test-cell-line GM \
        --cell-lines GM H1 HCT116 HepG2 IMR90 K562 MCF7 SKNSH \
        --chromatin-features H3K4me1 H3K4me2 H3K4me3 H3K9me3 RAD21 CTCF H3K36me3 H3K79me2 H3K27ac H3K9ac H3K27me3 H2AFZ H4K20me1 \
        --feature-groups 1 2 3 4 5 \
        --output-dir ./results
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
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

# --- Custom module imports (adjust if needed) ---
from data_preparation import prepare_interaction_data
from train_test_features import train_test_split_by_cell
from features_enginearing import (
    add_interaction_distance_and_strand_features,
    compile_chromatin_features,
    merge_conservation_with_interactions,
)
from neighboring_features import data_neighboring_features

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
    """
    Prepare the full dataset combining interaction, chromatin, and conservation features.
    """
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


def plot_feature_importance(feature_importance_df, top_n=20, title="Feature Importance", save_path=None):
    """Plot top N feature importances."""
    plt.figure(figsize=(10, 8))
    top_features = feature_importance_df.head(top_n)

    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.title(f'{title} - Top {top_n} Features', fontsize=14, fontweight='bold')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def save_feature_importance(feature_importance_df, output_path):
    """Save feature importance to CSV file."""
    feature_importance_df.to_csv(output_path, index=False)
    logging.info(f"✅ Feature importance saved to: {output_path}")


def evaluate_group_performance(y_true, y_pred, group_labels, dataset_name="Dataset"):
    """Evaluate and print performance metrics by group."""
    logging.info("\n" + "=" * 60)
    logging.info(f"PERFORMANCE BY GROUP ({dataset_name})")
    logging.info("=" * 60)

    for group in sorted(group_labels.unique()):
        mask = group_labels == group
        if mask.any():
            y_group = y_true[mask]
            pred_group = y_pred[mask]
            acc = accuracy_score(y_group, pred_group)
            p, r, f1, _ = precision_recall_fscore_support(
                y_group, pred_group, average='binary', zero_division=0
            )
            n_pos = (y_group == 1).sum()
            n_neg = (y_group == 0).sum()
            logging.info(
                f"Group {group}: Acc={acc:.4f}, Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}, "
                f"n={mask.sum()} (Pos={n_pos}, Neg={n_neg})"
            )


def train_and_evaluate_model(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    test_df,
    group_train, group_val, group_test,
    features_group,
    output_dir,
    drop_cols=None,
) -> tuple[pd.DataFrame, GradientBoostingClassifier, pd.DataFrame]:
    """
    Train a Gradient Boosting model and evaluate performance on the test set.
    Optionally drop specified columns.
    """
    logging.info("Training Gradient Boosting Classifier...")

    # Make copies to avoid modifying original data
    X_train_c = X_train.copy()
    X_val_c = X_val.copy()
    X_test_c = X_test.copy()
    test_df_c = test_df.copy()

    # Drop columns if specified
    if drop_cols is not None:
        cols_to_drop = [col for col in drop_cols if col in X_train_c.columns]
        if cols_to_drop:
            logging.info(f"Dropping columns: {cols_to_drop}")
            X_train_c = X_train_c.drop(columns=cols_to_drop)
            X_val_c = X_val_c.drop(columns=cols_to_drop)
            X_test_c = X_test_c.drop(columns=cols_to_drop)

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        min_samples_split=30,
        min_samples_leaf=15,
        random_state=42,
    )
    model.fit(X_train_c, y_train)

    # Predictions
    train_pred = model.predict(X_train_c)
    val_pred = model.predict(X_val_c)
    test_pred = model.predict(X_test_c)
    test_proba = model.predict_proba(X_test_c)[:, 1]

    # Metrics
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    logging.info("\n" + "=" * 60)
    logging.info("MODEL PERFORMANCE")
    logging.info("=" * 60)
    logging.info(f"Train Accuracy: {train_acc:.4f}")
    logging.info(f"Validation Accuracy: {val_acc:.4f}")
    logging.info(f"Test Accuracy: {test_acc:.4f}")
    logging.info(f"Train Positive Rate: {y_train.mean():.4f}")
    logging.info(f"Validation Positive Rate: {y_val.mean():.4f}")
    logging.info(f"Test Positive Rate: {y_test.mean():.4f}")

    logging.info("\n" + "=" * 60)
    logging.info("CLASSIFICATION REPORT - TRAIN SET")
    logging.info("=" * 60)
    logging.info("\n" + classification_report(y_train, train_pred))

    logging.info("\n" + "=" * 60)
    logging.info("CLASSIFICATION REPORT - VALIDATION SET")
    logging.info("=" * 60)
    logging.info("\n" + classification_report(y_val, val_pred))

    logging.info("\n" + "=" * 60)
    logging.info("CLASSIFICATION REPORT - TEST SET")
    logging.info("=" * 60)
    logging.info("\n" + classification_report(y_test, test_pred))

    evaluate_group_performance(y_train, train_pred, group_train, "TRAIN SET")
    evaluate_group_performance(y_val, val_pred, group_val, "VALIDATION SET")
    evaluate_group_performance(y_test, test_pred, group_test, "TEST SET")

    # Feature importance
    logging.info("\n" + "=" * 60)
    logging.info("FEATURE IMPORTANCE ANALYSIS")
    logging.info("=" * 60)

    feature_importance = pd.DataFrame({
        'feature': X_train_c.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Create output subdirectory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Save importance
    imp_path = os.path.join(output_dir, f'feature_importance_{features_group}.csv')
    save_feature_importance(feature_importance, imp_path)

    # Plot
    plot_feature_importance(
        feature_importance,
        top_n=20,
        title=f"Feature Importance - Group {features_group}",
        save_path=os.path.join(output_dir, f'feature_importance_plot_{features_group}.png'),
    )

    # Cumulative importance plot
    cum_imp = feature_importance['importance'].cumsum()
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cum_imp) + 1), cum_imp, 'b-', linewidth=2)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% cumulative importance')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance')
    plt.title(f'Cumulative Feature Importance - Group {features_group}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cumulative_importance_{features_group}.png'), dpi=300)
    plt.show()

    # Add predictions to test dataframe
    test_df_c["probability"] = test_proba
    test_df_c["prediction"] = test_pred

    return test_df_c, model, feature_importance


def apply_group_aware_sampling_train(X, y, groups, n_pos=10000, n_neg=10000):
    """
    Sample exactly n_pos positive and n_neg negative samples from each group.
    Groups with insufficient samples are removed.
    Returns sampled data and list of kept groups.
    """
    X_bal_list, y_bal_list, g_bal_list = [], [], []
    kept_groups = []

    for group in sorted(groups.unique()):
        mask = groups == group
        X_g = X[mask]
        y_g = y[mask]

        n_pos_actual = (y_g == 1).sum()
        n_neg_actual = (y_g == 0).sum()

        if n_pos_actual >= n_pos and n_neg_actual >= n_neg:
            pos_idx = y_g[y_g == 1].index
            neg_idx = y_g[y_g == 0].index
            sampled_pos = np.random.choice(pos_idx, size=n_pos, replace=False)
            sampled_neg = np.random.choice(neg_idx, size=n_neg, replace=False)
            keep_idx = list(sampled_pos) + list(sampled_neg)

            X_bal = X_g.loc[keep_idx]
            y_bal = y_g.loc[keep_idx]
            g_bal = pd.Series([group] * len(y_bal), index=y_bal.index)

            X_bal_list.append(X_bal)
            y_bal_list.append(y_bal)
            g_bal_list.append(g_bal)
            kept_groups.append(group)
            logging.info(f"  Group {group}: sampled {n_pos} pos + {n_neg} neg")
        else:
            logging.info(f"  Group {group}: REMOVED (pos={n_pos_actual}, neg={n_neg_actual})")

    if not X_bal_list:
        raise ValueError("No groups with sufficient samples.")

    X_bal = pd.concat(X_bal_list, axis=0)
    y_bal = pd.concat(y_bal_list, axis=0)
    g_bal = pd.concat(g_bal_list, axis=0)

    logging.info(f"Kept groups: {kept_groups}")
    logging.info(f"Final training size: {len(y_bal)} samples")
    return X_bal, y_bal, g_bal, kept_groups


def get_sampled_indices_test(y, groups, train_kept_groups, random_state=42):
    """
    For each group present in train_kept_groups, sample an equal number of
    positive and negative examples (minimum class size) from the test data.
    Returns the list of indices to keep.
    """
    np.random.seed(random_state)
    sampled_indices = []

    for group in train_kept_groups:
        if group not in groups.unique():
            continue

        mask = groups == group
        y_g = y[mask]
        idx_g = y[mask].index

        n_pos = (y_g == 1).sum()
        n_neg = (y_g == 0).sum()
        min_samples = min(n_pos, n_neg)

        if min_samples == 0:
            continue

        pos_idx = idx_g[y_g == 1].tolist()
        neg_idx = idx_g[y_g == 0].tolist()

        sampled_pos = np.random.choice(pos_idx, size=min_samples, replace=False)
        sampled_neg = np.random.choice(neg_idx, size=min_samples, replace=False)

        sampled_indices.extend(sampled_pos)
        sampled_indices.extend(sampled_neg)

    return sampled_indices


def compatibility(pairs_df):
    """Add label column based on PET counts."""
    pairs_df['all_int_cell'] = 8 - pairs_df['missing_interactions']

    positive_samples = []
    negative_samples = []
    cell_lines = pairs_df['cell_line'].unique()

    for cell_line in cell_lines:
        pos = pairs_df.query(f"cell_line == @cell_line and pet_{cell_line} > 0")
        neg = pairs_df.query(f"cell_line == @cell_line and pet_{cell_line} == 0")
        positive_samples.append(pos)
        negative_samples.append(neg)

    positive_samples = pd.concat(positive_samples)
    negative_samples = pd.concat(negative_samples)
    positive_samples["label"] = 1
    negative_samples["label"] = 0

    return pd.concat([positive_samples, negative_samples])


def group_specific(
    test_cell_line,
    features_group,
    cell_lines,
    chromatin_features,
    paths_dict,
    output_dir,
    output_dir_dropped,
):
    """Run group-specific model training and evaluation."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Processing: Test Cell={test_cell_line}, Group={features_group}")
    logging.info(f"{'='*60}\n")

    # Prepare all features
    pairs_df = prepare_all_features(
        paths_dict['chia_pet_dir'],
        paths_dict['chipseq_dir'],
        paths_dict['chromatin_dir'],
        paths_dict['rpkm_features_dir'],
        paths_dict['rpkm_intervals_dir'],
        paths_dict['intersect_motif_chia_dir'],
        paths_dict['ctcf_motifs_dir'],
        cell_lines,
        chromatin_features,
    )
    pairs_df = compatibility(pairs_df)

    # Neighbouring features
    neigh = data_neighboring_features(pairs_df, features_group)
    neigh = neigh.query('all_int_cell != @features_group')

    neigh2 = pd.DataFrame()
    for cell in cell_lines:
        df1 = pairs_df.query('cell_line==@cell')[['sequence_name', 'start1', 'stop2']]
        df2 = pairs_df.query(
            f'cell_line==@cell and pet_{cell}>0 and all_int_cell==@features_group'
        )[['sequence_name', 'start1', 'stop2']]

        if df1.empty or df2.empty:
            logging.info(f"  Cell {cell}: no data for overlap")
            continue

        overlap = fast_overlaps_with_percentage(df1, df2)
        if overlap.empty:
            logging.info(f"  Cell {cell}: no overlaps")
            continue

        overlap['max_overlap'] = overlap[['percent_of_df1', 'percent_of_df2']].max(axis=1)
        overlap = overlap.query('percent_of_df1!=100 or percent_of_df2!=100')
        if overlap.empty:
            continue

        in_loop = overlap.query('percent_of_df1>95').groupby(
            ['sequence_name', 'df1_start', 'df1_stop']
        ).size().to_frame('inerr_loop').reset_index()
        out_loop = overlap.query('percent_of_df2>95').groupby(
            ['sequence_name', 'df1_start', 'df1_stop']
        ).size().to_frame('outer_loop').reset_index()
        cross = overlap.query('max_overlap<95').groupby(
            ['sequence_name', 'df1_start', 'df1_stop']
        ).size().to_frame('cross').reset_index()

        merged = in_loop.merge(out_loop, how='outer').fillna(0)
        merged = merged.merge(cross, how='outer').fillna(0)
        merged = merged.rename({'df1_start': 'start1', 'df1_stop': 'stop2'}, axis=1)
        merged['cell_line'] = cell
        neigh2 = pd.concat([neigh2, merged])

    neigh = neigh.merge(neigh2, how='left').fillna(0)

    neigh_cols = list(set(neigh.columns) - set(pairs_df.columns))
    neigh_cols = ["sequence_name", "start1", "stop1", "start2", "stop2", "cell_line", "all_int_cell"] + neigh_cols

    # Train/test split by cell
    train_df, test_df = train_test_split_by_cell(
        pairs_df,
        cell_lines,
        test_cell_line,
        chromatin_features,
    )
    train_df = train_df.drop(columns=['all_int_cell'])
    test_df = test_df.drop(columns=['all_int_cell'])

    train_df = train_df.merge(neigh[neigh_cols])
    test_df = test_df.merge(neigh[neigh_cols])
    test_df = test_df.query('all_int_cell > 0')

    group_train = train_df['all_int_cell'].copy()
    group_test = test_df['all_int_cell'].copy()

    logging.info(f"Train groups: {group_train.unique()}")
    logging.info(f"Test groups: {group_test.unique()}")
    logging.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    NON_PREDICTORS = [
        "sequence_name", "start1", "stop1", "start2", "stop2",
        "label", "cell_line", "all_int_cell", "ave_pet", "std_pet",
        "ave_degree1", "ave_degree2", "std_degree1", "std_degree2"
    ]

    X_train = train_df.drop(columns=NON_PREDICTORS)
    y_train = train_df["label"]
    X_test = test_df.drop(columns=NON_PREDICTORS)
    y_test = test_df["label"]

    # Sampling
    logging.info("\n" + "=" * 60)
    logging.info("APPLYING SMART SAMPLING (Preserving Group Features)")
    logging.info("=" * 60)

    X_bal, y_bal, groups_bal, kept_groups = apply_group_aware_sampling_train(
        X_train, y_train, group_train
    )

    # Train/validation split
    X_tr, X_val, y_tr, y_val, g_tr, g_val = train_test_split(
        X_bal, y_bal, groups_bal,
        test_size=0.2,
        random_state=42,
        stratify=groups_bal,
    )

    # Sample test data using kept groups
    sampled_idx = get_sampled_indices_test(y_test, group_test, kept_groups)
    test_df_sampled = test_df.loc[sampled_idx].reset_index(drop=True)
    X_test_bal = test_df_sampled.drop(columns=NON_PREDICTORS)
    y_test_bal = test_df_sampled["label"]
    g_test_bal = group_test.loc[sampled_idx].reset_index(drop=True)

    logging.info(f"Original test size: {len(test_df)}, Sampled test size: {len(test_df_sampled)}")

    # Model 1: without dropping
    test_out, _, _ = train_and_evaluate_model(
        X_tr, y_tr,
        X_val, y_val,
        X_test_bal, y_test_bal,
        test_df_sampled,
        g_tr, g_val, g_test_bal,
        features_group,
        output_dir,
        drop_cols=None,
    )
    test_out.to_csv(os.path.join(output_dir, f'grouping_{test_cell_line}_{features_group}.csv'), index=False)

    # Model 2: drop neighbour columns
    test_out_dropped, _, _ = train_and_evaluate_model(
        X_tr, y_tr,
        X_val, y_val,
        X_test_bal, y_test_bal,
        test_df_sampled,
        g_tr, g_val, g_test_bal,
        features_group,
        output_dir_dropped,
        drop_cols=neigh_cols,
    )
    test_out_dropped.to_csv(os.path.join(output_dir_dropped, f'grouping_{test_cell_line}_{features_group}.csv'), index=False)

    return test_out, test_out_dropped


def aggregate_feature_importance_across_groups(feature_groups, output_dir, prefix=""):
    """Aggregate feature importance across all groups and create summary."""
    all_importances = []

    for group in feature_groups:
        fname = os.path.join(output_dir, f'feature_importance_{group}.csv')
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            df['group'] = group
            all_importances.append(df)
        else:
            logging.warning(f"Missing importance file: {fname}")

    if not all_importances:
        logging.warning("No importance files found.")
        return None

    combined = pd.concat(all_importances, ignore_index=True)
    avg_imp = combined.groupby('feature')['importance'].agg(['mean', 'std', 'count']).reset_index()
    avg_imp = avg_imp.sort_values('mean', ascending=False)

    out_csv = os.path.join(output_dir, f'aggregated_feature_importance_{prefix}.csv')
    avg_imp.to_csv(out_csv, index=False)

    # Plot top 20
    plt.figure(figsize=(12, 10))
    top20 = avg_imp.head(20)
    sns.barplot(data=top20, y='feature', x='mean', palette='viridis')
    plt.title(f'Average Feature Importance Across Groups (Top 20) - {prefix}')
    plt.xlabel('Mean Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'aggregated_importance_plot_{prefix}.png'), dpi=300)
    plt.show()

    logging.info("\n" + "=" * 60)
    logging.info(f"TOP 20 FEATURES (AVERAGE) - {prefix}")
    logging.info("=" * 60)
    for idx, row in top20.iterrows():
        logging.info(f"{idx+1:2d}. {row['feature']:40s} {row['mean']:.6f} (±{row['std']:.6f})")

    return avg_imp


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train group-specific chromatin interaction models.")
    parser.add_argument('--chia-pet-dir', required=True, help="Directory with ChIA-PET interaction files")
    parser.add_argument('--chipseq-dir', required=True, help="Directory with ChIP-seq peak files")
    parser.add_argument('--chromatin-dir', required=True, help="Directory with chromatin feature files")
    parser.add_argument('--rpkm-features-dir', required=True, help="Directory with RPKM features")
    parser.add_argument('--rpkm-intervals-dir', required=True, help="Directory with RPKM intervals")
    parser.add_argument('--intersect-motif-chia-dir', required=True, help="Directory with motif-ChIA intersection")
    parser.add_argument('--ctcf-motifs-dir', required=True, help="Directory with CTCF motif files")
    parser.add_argument('--test-cell-line', default='GM', help="Cell line to use as test (default: GM)")
    parser.add_argument('--cell-lines', nargs='+',
                        default=['GM', 'H1', 'HCT116', 'HepG2', 'IMR90', 'K562', 'MCF7', 'SKNSH'],
                        help="List of cell lines")
    parser.add_argument('--chromatin-features', nargs='+',
                        default=["H3K4me1", "H3K4me2", "H3K4me3", "H3K9me3", "RAD21", "CTCF",
                                 "H3K36me3", "H3K79me2", "H3K27ac", "H3K9ac", "H3K27me3", "H2AFZ", "H4K20me1"],
                        help="List of chromatin features")
    parser.add_argument('--feature-groups', nargs='+', type=int, default=list(range(1, 9)),
                        help="Feature group numbers to process (e.g., 1 2 3 4 5)")
    parser.add_argument('--output-dir', default='./results', help="Base output directory (default: ./results)")
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Create output directories
    out_all = os.path.join(args.output_dir, 'all_features')
    out_dropped = os.path.join(args.output_dir, 'dropped_neighbour')
    os.makedirs(out_all, exist_ok=True)
    os.makedirs(out_dropped, exist_ok=True)

    paths = {
        'chia_pet_dir': args.chia_pet_dir,
        'chipseq_dir': args.chipseq_dir,
        'chromatin_dir': args.chromatin_dir,
        'rpkm_features_dir': args.rpkm_features_dir,
        'rpkm_intervals_dir': args.rpkm_intervals_dir,
        'intersect_motif_chia_dir': args.intersect_motif_chia_dir,
        'ctcf_motifs_dir': args.ctcf_motifs_dir,
    }

    # Process each feature group
    for g in args.feature_groups:
        group_specific(
            args.test_cell_line,
            g,
            args.cell_lines,
            args.chromatin_features,
            paths,
            out_all,
            out_dropped,
        )

    # Aggregate and summarise
    logging.info("\n" + "=" * 60)
    logging.info("AGGREGATING FEATURE IMPORTANCE - ALL FEATURES")
    logging.info("=" * 60)
    aggregate_feature_importance_across_groups(args.feature_groups, out_all, prefix="all")

    logging.info("\n" + "=" * 60)
    logging.info("AGGREGATING FEATURE IMPORTANCE - DROPPED NEIGHBOUR")
    logging.info("=" * 60)
    aggregate_feature_importance_across_groups(args.feature_groups, out_dropped, prefix="dropped")

    logging.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
