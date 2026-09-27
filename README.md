# NIDS — Network Intrusion Detection System (NSL-KDD)

Network intrusion detection using Machine Learning on the **NSL-KDD** dataset. The project compares four classification algorithms to distinguish normal traffic from attacks (DoS, Probe, R2L, U2R), with model explainability using SHAP.

## Context

A Network Intrusion Detection System (NIDS) analyzes network traffic to automatically identify malicious connections. This project uses **NSL-KDD**, a refined version of the KDD Cup 99 dataset (duplicates removed, more realistic class distribution), containing network connections labeled with 41 features (duration, protocol, bytes transferred, error rates, etc.).

## Dataset

Source: [NSL-KDD Dataset (GitHub)](https://github.com/jmnwong/NSL-KDD-Dataset) — text version from the [University of New Brunswick](https://www.unb.ca/cic/datasets/nsl.html)

| File | Rows | Description |
|---|---|---|
| `KDDTrain+.txt` | 125,973 | Training set |
| `KDDTest+.txt` | 22,544 | Test set (includes attack types absent from training, by design) |

**Attack categories**: `normal`, `dos` (Denial of Service), `probe` (reconnaissance), `r2l` (Remote to Local), `u2r` (User to Root). The 40 raw attack labels (`neptune`, `satan`, `apache2`, `saint`, `snmpguess`, etc.) were mapped to these 5 categories based on their documented attack type.

## Methodology

1. **Loading & cleaning** — column naming, missing-value/duplicate checks, label grouping into 5 categories.
2. **Feature engineering** — One-Hot Encoding of categorical variables (`protocol_type`, `service`, `flag`), column alignment between train/test (6 rare `service` values present only in train), feature/target split.
3. **Modeling** — four models trained and compared: Random Forest, XGBoost, SVM (RBF kernel, trained on a 20,000-row stratified subsample for tractability), and a feed-forward Neural Network (Keras, 64→32→5, dropout 0.3, 20 epochs). SVM and the Neural Network required feature standardization (`StandardScaler`); tree-based models did not.
4. **Class imbalance correction** — `sample_weight` (`class_weight="balanced"`) tested on XGBoost.
5. **Explainability** — SHAP (`TreeExplainer`) on the final XGBoost model, on a 500-row test subsample.

## Results

| Model | Accuracy | F1 macro avg | Notes |
|---|---|---|---|
| Random Forest | 0.751 | 0.48 | Fast baseline; near-zero recall on `r2l`/`u2r` |
| **XGBoost** | 0.774 | **0.57** | Best balance across classes; selected as final model |
| SVM | 0.741 | 0.51 | Trained on a 20k-row subsample only |
| Neural Network (Keras) | **0.786** | 0.56 | Best raw accuracy; high precision but low recall on `r2l`/`u2r` |

**Per-class F1-score (XGBoost, final model):**

| Category | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| dos | 0.96 | 0.83 | 0.89 | 7,458 |
| normal | 0.68 | 0.97 | 0.80 | 9,711 |
| probe | 0.80 | 0.63 | 0.71 | 2,421 |
| r2l | 0.98 | 0.11 | 0.20 | 2,885 |
| u2r | 0.69 | 0.16 | 0.26 | 69 |

**Class imbalance correction (XGBoost, `class_weight="balanced"`):** improved `probe` (F1 0.71→0.76) but did not resolve the `r2l` weakness (F1 0.20→0.15) — recall on `r2l` remained low, suggesting the issue is not purely a sample-size imbalance: the test set's `r2l` attacks differ structurally from the ones seen during training (different sub-techniques), which reweighting cannot fix on its own.

## Explainability (SHAP)

SHAP analysis of the final XGBoost model shows that, for the `dos` class, `src_bytes` (low byte counts, consistent with incomplete SYN-flood-style connections), `same_srv_rate`, and `flag_S0` (incomplete TCP handshake) are the most influential features — consistent with the known signature of denial-of-service attacks (see `results/shap_summary_dos.png`).

## Key limitations

- SVM was trained on a 20,000-row subsample rather than the full training set, for computational tractability — comparison with the other models trained on the full 125,973 rows is not perfectly apples-to-apples.
- `r2l` and `u2r` remain difficult to detect across all models due to their rarity in training data (995 and 52 samples respectively) and structural differences between train/test attack sub-types.
- TensorFlow could not initially be installed system-wide due to a Windows long-path limitation; resolved using a Python virtual environment (`.venv`).

## Installation

```bash
git clone https://github.com/emnathouabtia/nids-nsl-kdd.git
cd nids-nsl-kdd
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Author

Emna Thouabtia — Computer Engineering Student, ENICarthage
