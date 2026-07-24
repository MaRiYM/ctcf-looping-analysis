# Exploring the Influence of Neighboring Interactions on CTCF-Mediated Chromatin Loop Formation

This repository contains the code and analysis used in the paper:

> **Exploring the Influence of Neighboring Interactions on CTCF-Mediated Chromatin Loop Formation**  
> *Maryam Mirabolghasemi et al., 2025*

---

## 🧬 Overview

This project explores how neighboring chromatin interactions influence CTCF-mediated loop formation using computational modeling and data analysis.  
It includes scripts for data preprocessing, feature extraction, machine learning analysis, and visualization of chromatin looping patterns.
This repository contains **two machine learning pipelines** for predicting CTCF-mediated chromatin interactions:


The pipeline performs the following steps:

1. **Data Preparation** – Loads and processes ChIA-PET and ChIP-Seq data for each cell line.  
2. **Feature Engineering** – Adds interaction distance, strand, conservation, and chromatin features.  
3. **Neighboring Feature Computation** – Aggregates features from neighboring genomic regions.  
4. **Model Training & Evaluation** – Trains a Gradient Boosting model and evaluates on a held-out test cell line.  
5. **Visualization** – Displays the confusion matrix and model performance metrics.

---

## ⚙️ Installation

### 1. Clone this repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create and activate a virtual environment (recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:
```
pandas
scikit-learn
numpy
matplotlib
```

---
📊 Input Data

This project uses several large datasets (ChIA-PET, ChIP-Seq, RPKM, and conservation data).
Because these files are large, they are not included in the repository.

Please download them manually from the following link(s):

🔗 https://drive.google.com/drive/folders/1qoFvcR3oNQdQ6UgsIXYOrKjXebxs6CQK?usp=drive_link


## 🚀 Usage

Run **one** of the two available models from the command line.  
Replace `PATH` with the actual folder locations (relative or absolute).

### 1. 2step_model.py – computed neighboring features

  ```bash
python two_step_pipeline.py \
    --chia-pet-dir ./data/intersect_chiapet \
    --chipseq-dir ./data/intersect_chipseq \
    --chromatin-dir ./data/motif_chromatin_features \
    --rpkm-features-dir ./data/rpkm_features \
    --rpkm-intervals-dir ./data/rpkm_intervals \
    --intersect-motif-chia-dir ./data/intersect_motif_chia \
    --ctcf-motifs-dir ./data/ctcf_motifs \
    --test-cell-line K562 \
    --cell-lines GM H1 HCT116 HepG2 IMR90 K562 MCF7 SKNSH \
    --output-dir ./my_results
  ```

### 2. Group-Specific_model.py – Data-driven neighboring features
  
  ```bash
python run_pipeline.py \
    --chia-pet-dir /path/to/intersect_chiapet \
    --chipseq-dir /path/to/intersect_chipseq \
    --chromatin-dir /path/to/motif_chromatin_features \
    --rpkm-features-dir /path/to/rpkm_features \
    --rpkm-intervals-dir /path/to/rpkm_intervals \
    --intersect-motif-chia-dir /path/to/intersect_motif_chia \
    --ctcf-motifs-dir /path/to/ctcf_motifs \
    --test-cell-line GM \
    --cell-lines GM H1 HCT116 HepG2 IMR90 K562 MCF7 SKNSH \
    --chromatin-features H3K4me1 H3K4me2 H3K4me3 H3K9me3 RAD21 CTCF H3K36me3 H3K79me2 H3K27ac H3K9ac H3K27me3 H2AFZ H4K20me1 \
    --feature-groups 1 2 3 4 5 \
    --output-dir ./results \
    --log-level INFO
  ```

---



## 📊 Output


./grouping/                                 # Full-feature model results
│   ├── feature_importance_3.csv            # Per-feature importance scores
│   ├── feature_importance_plot_3.png       # Top-20 feature importance bar chart
│   ├── cumulative_importance_3.png         # Cumulative importance curve
│   └── grouping_HCT116_3.csv               # Test set predictions + probabilities
│
./grouping0/                                # Ablated model (neighbouring features dropped)
│   ├── feature_importance_3.csv
│   ├── feature_importance_plot_3.png
│   ├── cumulative_importance_3.png
│   └── grouping_HCT116_3.csv
│
./grouping/aggregated_feature_importance.csv # (If multiple groups are run) Average importance across groups


## ⚗️ Parameters

| Parameter | Description | Example |
|------------|-------------|----------|
| `--motif_chromatin_features` | Path to motif chromatin features directory | `"D:/.../motif_chromatin_features"` |
| `--intersect_chiapet` | Directory with ChIA-PET motif intersections | `"D:/.../intersect_chiapet"` |
| `--intersect_chipseq` | Directory with ChIP-Seq motif intersections | `"D:/.../intersect_chipseq"` |
| `--rpkm_features` | Path to RPKM feature outputs | `"D:/.../rpkm_output"` |
| `--rpkm_intervals` | Path to RPKM intervals | `"D:/.../features/rpkm_intervals"` |
| `--intersect_motif_chia` | Path to overlaps between motif and ChIA-PET | `"D:/.../intersect_motif_chia"` |
| `--ctcf_motifs` | Path to CTCF motif file | `"D:/.../CTCF_Motifs"` |
| `--cell_lines` | List of cell lines used in training | `"GM,H1,HCT116,HepG2"` |
| `--test_cell_line` | Cell line used for testing | `"HCT116"` |
| `--features_group` | Number of neighboring groups | `3` |

---


## 📄 License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute it for research or educational purposes.

---

## 👩‍🔬 Author

**<Maryam Mirabolghasemi>**  
PhD Student in Bioinformatics  
📧 _<M.mirabolghasemi@ut.ac.ir>_  

---

## 🌟 Acknowledgements

This project integrates computational and bioinformatics techniques to study **chromatin organization** and **CTCF-mediated interactions**, combining expertise in genomics, data science, and machine learning.


