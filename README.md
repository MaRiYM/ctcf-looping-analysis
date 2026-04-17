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
  python 2step_model.py \
  --motif_chromatin     "data/motif_chromatin path" \
  --intersect_chiapet   "data/chiapet_intersect path" \
  --intersect_chipseq   "data/chipseq_intersect path" \
  --rpkm_features       "data/rpkm_features path" \
  --rpkm_intervals      "data/rpkm_intervals path" \
  --intersect_motif_chia "data/motif_chia_intersect path" \
  --ctcf_motifs         "data/motif_chromatin path" \
  --cell_lines "GM H1 HCT116 HepG2 IMR90 K562 MCF7 SKNSH" \
  --test_cell HCT116
  ```

### 2. Group-Specific_model.py – Data-driven neighboring features
  
  ```bash
  python Group-Specific_model.py \
  --motif_chromatin_features "data/motif_chromatin_features path" \
  --intersect_chiapet        "data/chiapet_intersect path" \
  --intersect_chipseq        "data/chipseq_intersect path" \
  --rpkm_features            "data/rpkm_features path" \
  --rpkm_intervals           "data/rpkm_intervals path" \
  --intersect_motif_chia     "data/motif_chia_intersect path" \
  --ctcf_motifs              "data/motif_chromatin_features path" \
  --cell_lines "GM,H1,HCT116,HepG2,IMR90,K562,MCF7,SKNSH" \
  --test_cell_line "HCT116" \
  --features_group 3
  ```

---



## 📊 Output

- Prints progress of each data preparation and feature engineering step  
- Displays **model accuracy** and **confusion matrix**
- Produces:
  - `train_neighboring_df` → Neighboring features for training data  
  - `test_neighboring_df` → Neighboring features for test data  
  - `test_confidence_df` → Test predictions with probabilities  

Example log:
```
Step 1: Preparing motif interaction data...
Step 2: Adding interaction distance and strand features...
Step 3: Compiling chromatin features...
Step 4: Merging conservation features...
✅ Feature preparation complete.
✅ Accuracy: 0.8732
```

---

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


