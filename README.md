# Data and Feature Analysis

This folder contains the processed datasets and statistical outputs used in the paper:

**“Head Directionality and Dependency Marking in Middle Persian Nominal Phrases: Quantitative Evidence from Ezafe Constructions.”**

---

## Folder Structure

```
.
├── data/
│   ├── preprocessed/
│   │   └── head_modifier_pairs.csv   # Final dataset (N = 8,019)
│   └── external/
│       └── MPCD_link.txt             # Link to the raw Zoroastrian MPCD corpus
└── artifacts/
    └── feature_analysis/             # Statistical interpretative outputs
        ├── per_class_importance_with_direction.csv
        ├── top_features_[Class_Name].csv
        ├── comparative_table.csv
        ├── spearman_correlation.csv
        └── full_normalized_importances.csv
```

---

## 1. Data (`data/`)

### `preprocessed/head_modifier_pairs.csv`

The primary dataset extracted from the MPCD. It contains **8,019 annotated pairs**. This file is used by the Random Forest model to achieve the reported **82.12% accuracy**.

**Key columns include:**

* `head_upos`: Universal POS tag of the head
* `ezafe_label`: Binary (0 = No Ezafe, 1 = With Ezafe)
* `position`: Linear order (1 = Head Initial, 2 = Head Final)
* `np_depth`: The hierarchical nesting depth of the phrase (primary complexity metric)
* `num_dependents_modifier`: Internal phrasal weight of the modifier

### `external/MPCD_link.txt`

To comply with corpus licensing, the raw manuscripts are not included. This file provides instructions and official links to the **Zoroastrian Middle Persian Corpus and Dictionary (MPCD)** for users who wish to access the source texts.

---

## 2. Feature Analysis (`artifacts/feature_analysis/`)

These files contain the mathematical decomposition of the model’s logic, providing the evidence for the **Functional Selection** argument in the paper.

* `per_class_importance_with_direction.csv`
  The master results file. Lists every feature, its importance score per class, and the SHAP direction (Positive/Negative). This is the source for **Table 6**.

* `top_features_[Class_Name].csv`
  Four files isolating the most predictive features for each morphosyntactic configuration (e.g., *With Ezafe & Head Initial*).

* `spearman_correlation.csv`
  A correlation matrix showing relationships between structural features (e.g., how `np_depth` correlates with distance).

* `comparative_table.csv`
  A summary table comparing how predictors change importance across structural environments.

* `full_normalized_importances.csv`
  Global ranking of all Boruta-selected features, normalized for cross-model comparison (source for **Figure 5**).

---

## Usage for Reproduction

To verify the linguistic thresholds (Section 4.2 of the paper):

1. Use the `visualization.ipynb` notebook to load these CSV files.
2. The model’s **positive direction** for `np_depth` in the *With Ezafe* class supports the **Complexity Principle**.
3. High importance of the `has_anchor` feature in the *Head Final* configuration supports the **Anchoring Hypothesis**.

---

## Citation

If you use these processed data or analysis tables in your research, please cite:

```bibtex
@article{musavi2026ezafe,
  title={Head Directionality and Dependency Marking in Nominal Phrases: Quantitative Evidence from Middle Persian Ezafe Constructions},
  author={Musavi, Seyyedeh Fatemeh (Raha)},
  year={2026},
  journal={Journal of Corpus Linguistics}
}
```
