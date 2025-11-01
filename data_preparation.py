import pandas as pd


def get_motif_interactions(
    chia_pet_dir: str,
    chipseq_dir: str,
    cell_line: str,
    replicate_id: int
) -> pd.DataFrame:
    """
    Extract motif interactions supported by both ChIA-PET and ChIP-Seq data.

    Parameters
    ----------
    chia_pet_dir : str
        Path to the directory containing intersections between CTCF motifs and ChIA-PET interactions.
    chipseq_dir : str
        Path to the directory containing intersections between CTCF motifs and ChIP-Seq data.
    cell_line : str
        Name of the cell line (e.g., 'HeLa', 'GM12878').
    replicate_id : int
        Replicate number of the ChIA-PET experiment (e.g., 1 or 2).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing motif pair interactions and their PET scores
        that are supported by ChIP-Seq data for the given cell line and replicate.
    """
    # Load intersections between CTCF motifs and ChIA-PET interactions
    interactions = (
        pd.read_csv(f"{chia_pet_dir}/{cell_line}_motif_rel_rep{replicate_id}.csv")[
            ["sequence_name1", "start1.1", "stop1", "start2.1", "stop2", "pet"]
        ]
        .sort_values("pet", ascending=False)
        .drop_duplicates(["sequence_name1", "start1.1", "stop1", "start2.1", "stop2"])
    )

    # Load intersections between CTCF motifs and ChIP-Seq data
    chipseq_motifs = pd.read_csv(f"{chipseq_dir}/CTCT_motif_{cell_line}.csv")

    # Keep only loops that have ChIP-Seq support
    cols_to_drop = chipseq_motifs.columns.tolist()
    interactions = interactions.merge(
        chipseq_motifs,
        left_on=["sequence_name1", "start1.1", "stop1"],
        right_on=["sequence_name", "start", "stop"],
    ).drop(cols_to_drop, axis=1)

    interactions = interactions.merge(
        chipseq_motifs,
        left_on=["sequence_name1", "start2.1", "stop2"],
        right_on=["sequence_name", "start", "stop"],
    ).drop(cols_to_drop, axis=1)

    # Keep the strongest PET-supported interactions
    interactions = (
        interactions[
            ["sequence_name1", "start1.1", "stop1", "start2.1", "stop2", "pet"]
        ]
        .sort_values("pet", ascending=False)
        .drop_duplicates(["sequence_name1", "start1.1", "stop1", "start2.1", "stop2"])
    )

    return interactions


import pandas as pd

def prepare_interaction_data(
    chia_pet_dir: str,
    chipseq_dir: str,
    cell_lines: list[str],
) -> pd.DataFrame:
    """
    Combine motif pair interactions and PET scores across multiple cell lines.

    Parameters
    ----------
    chia_pet_dir : str
        Directory containing intersections between CTCF motifs and ChIA-PET interactions.
    chipseq_dir : str
        Directory containing intersections between CTCF motifs and ChIP-Seq data.
    cell_lines : list[str]
        List of cell line names to include in the analysis.

    Returns
    -------
    pd.DataFrame
        A merged DataFrame containing motif pair interactions and their PET scores
        across all specified cell lines. Also includes a column counting how many
        cell lines have no interaction (NaN).
    """
    # Make a copy to avoid modifying the original list
    cell_lines_copy = list(cell_lines)

    # Use the first cell line as the base
    first_cell = cell_lines_copy[0]

    # Combine two replicates for the first cell line
    rep1 = get_motif_interactions(chia_pet_dir, chipseq_dir, first_cell, 1)
    rep2 = get_motif_interactions(chia_pet_dir, chipseq_dir, first_cell, 2)

    combined = rep1.merge(
        rep2, on=["sequence_name1", "start1.1", "stop1", "start2.1", "stop2"]
    )
    combined["pet"] = combined[["pet_x", "pet_y"]].max(axis=1)
    combined = (
        combined.rename(columns={"pet": f"pet_{first_cell}"})
        .drop(columns=["pet_x", "pet_y"])
    )

    # Process the remaining cell lines (excluding the first)
    for cell_line in cell_lines_copy[1:]:
        rep1 = get_motif_interactions(chia_pet_dir, chipseq_dir, cell_line, 1)
        rep2 = get_motif_interactions(chia_pet_dir, chipseq_dir, cell_line, 2)

        merged_rep = rep1.merge(
            rep2, on=["sequence_name1", "start1.1", "stop1", "start2.1", "stop2"]
        )
        merged_rep["pet"] = merged_rep[["pet_x", "pet_y"]].max(axis=1)
        merged_rep = (
            merged_rep.rename(columns={"pet": f"pet_{cell_line}"})
            .drop(columns=["pet_x", "pet_y"])
        )

        combined = combined.merge(
            merged_rep,
            on=["sequence_name1", "start1.1", "stop1", "start2.1", "stop2"],
            how="outer",
        )

    # Count missing (non-interacting) PET values across all cell lines
    combined["missing_interactions"] = combined.isna().sum(axis=1)

    # Replace NaN with 0 for PET values and rename columns
    combined = (
        combined.fillna(0)
        .rename(
            columns={
                "sequence_name1": "sequence_name",
                "start1.1": "start1",
                "start2.1": "start2",
            }
        )
    )

    return combined

