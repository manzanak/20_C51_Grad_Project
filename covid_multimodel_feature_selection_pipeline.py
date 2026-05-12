"""
=============================================================================
COVID-19 Status Classification from Placental Bulk RNA-seq Gene Expression
Pipeline: Nested CV + Multi-Model Stability Selection + Comprehensive Visualization
=============================================================================
Dataset : GSE318446 AnnData (n=65 samples, 19,702 genes)
Platform: Illumina NovaSeq X — bulk rRNA-depleted RNA-seq
          Kallisto v0.48.0 → GRCh38 Ensembl 110 (protein-coding)
Labels  : covid_status ("COVID +" / "COVID -"), binary
          fetal_sex used as key covariate for sex-stratified evaluation
Cohort  : 39 COVID+ / 26 COVID-, 33 Male / 32 Female

Pipeline Overview
-----------------
STEP 1: Load data → STEP 2: Global variance filter (top 50 genes for multimodel)
        ↓
STEP 3: Nested CV (5×5 folds) → Baseline AUC + out-of-fold predictions
        ↓
STEP 4: Multi-Model Stability Selection (5 algorithms × 100 bootstraps)
        ↓
STEP 5: Cross-Panel Portability Evaluation (test each panel via ElasticNet)
        ↓
STEP 6: Final Model on ElasticNet Stable Panel (4 genes)
        ↓
STEP 7: Comprehensive Visualization (14 figures)

CSV Outputs
-----------
  stable_gene_panel.csv                    4-gene panel + ElasticNet coefficients
  panel_portability_comparison.csv         AUC/size for all 5 model panels
  elasticnet_lr_stability_all_genes.csv    All genes ranked by bootstrap freq
  [model]_stable_gene_panel.csv            Stable genes for each of 5 models

Figure Outputs (14 total)
--------------------------
NESTED CV BASELINE (6 figures)
  fig3_roc_pr_curves.png                   ROC + Precision-Recall (OOF)
  fig4_sex_stratified_check.png            ROC curves by fetal sex
  fig8_gene_contributions_heatmap.png      Top 30 genes across CV folds
  fig9_sex_stratified_performance.png      AUC + Accuracy by sex
  fig11_confusion_matrix.png               TP/TN/FP/FN (OOF predictions)

MULTI-MODEL COMPARISON (6 figures — 5 algorithms tested)
  fig1_stability_barplot.png               Top 30 genes by ElasticNet frequency
  fig5_model_panel_sizes.png               Panel sizes across models
  fig6_model_agreement_heatmap.png         Gene overlap/agreement heatmap
  fig7_panel_performance.png               Cross-panel portability AUC
  fig12_model_feature_importance.png       Top genes: ElasticNet vs other models
  fig14_panel_roc_comparison.png           ROC curves using different panels

DATA EXPLORATION (1 figure)
  fig10_gene_expression_boxplots.png       Top 10 genes: expression by COVID status

FINAL MODEL (1 figure — 4-gene stable panel)
  fig2_coefficient_forest.png              Coefficients of stable panel
  fig13_final_model_sex_stratified_roc.png Sex-stratified ROC (stable panel)
=============================================================================
"""

# ── USER CONFIGURATION ────────────────────────────────────────────────────────
ANNDATA_PATH     = "/Users/kmanzana/Documents/20.C51_Project/COVID_Placenta/GSE318446_placenta_covid.h5ad"
COVID_COL        = "covid_status"       # values: "COVID +" / "COVID -"
SEX_COL          = "fetal_sex"          # values: "Male" / "Female"
COVID_POS_LABEL  = "COVID +"            # positive class label (exact string)
SYMBOL_COL       = "symbol"             # adata.var column for gene symbols
ENSEMBL_COL      = "ensembl_gene"       # adata.var column for Ensembl IDs

TOP_N_GENES      = 3000    # variance filter: keep top N most variable genes
N_BOOTSTRAP      = 100     # stability selection resamples
STABILITY_THRESH = 0.50    # min bootstrap frequency to call a gene "stable"
RANDOM_STATE     = 42

# ElasticNet hyperparameter grid
L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 1.0]   # 1.0 = pure LASSO
C_VALUES  = [10**x for x in [-3, -2.5, -2, -1.5, -1,
                               -0.5, 0, 0.5, 1, 1.5, 2]]  # 11 values

import datetime
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"covid_classification_results_{TIMESTAMP}"
# ─────────────────────────────────────────────────────────────────────────────

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = ['Arial', 'Microsoft sans serif']
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks", font="Arial")
import scanpy as sc

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.metrics import (roc_auc_score, roc_curve,
                             average_precision_score,
                             precision_recall_curve)
from scipy.stats import median_abs_deviation

warnings.filterwarnings("ignore")
os.makedirs(OUT_DIR, exist_ok=True)

# Colorblind-friendly palette
COL_COVID_POS = "#FF9400"   # COVID+
COL_COVID_NEG = "#0279EE"   # COVID-
COL_STABLE    = "#75A025"
COL_UNSTABLE  = "#ECE9E2"
COL_FEMALE    = "#FD9BED"
COL_MALE      = "#0279EE"

MODEL_COLORS = {
    "ElasticNet LR":      "#0279EE",
    "Random Forest":      "#FF9400",
    "SVM (RBF)":          "#75A025",
    "Gradient Boosting":  "#E63946",
    "k-NN":               "#9B5DE5",
}


# =============================================================================
# STEP 1 — Load AnnData
# =============================================================================
def load_data(path, covid_col, sex_col, covid_pos_label):
    import anndata as ad
    print(f"Loading: {path}")
    adata = ad.read_h5ad(path)
    print(adata)

    # Filters genes expressed in fewer than 26 samples (40% of 65) to reduce noise and speed up ML.
    sc.pp.filter_genes(adata, min_cells=26)
    print(f"\nAfter filtering low-expression genes (expressed in <40% of samples):")
    print(f"  X_raw shape: {adata.shape}  (samples × genes)")

    # Transform to log1p(TPM) for ML (compresses dynamic range, makes distribution more Gaussian).
    sc.pp.log1p(adata)  # In-place log1p transformation of adata.X
    print(f"\nApplied log1p transformation to adata.X for ML.")
    print(f"  X range: [{adata.X.min():.2f}, {adata.X.max():.2f}]")

    # ── Expression matrix ─────────────────────────────────────────────────────
    
    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    X = X.astype(np.float64)


    # X contains TPM values → apply log1p for ML (compresses dynamic range,
    # makes distribution more Gaussian, standard practice for RNA-seq ML)
    # X     = np.log1p(X_raw)   # log1p(TPM)
    # print(f"\nExpression: log1p(TPM) applied")
    # print(f"  X shape: {X.shape}  (samples × genes)")
    # print(f"  X range: [{X.min():.2f}, {X.max():.2f}]")

    
    # ── Gene identifiers ──────────────────────────────────────────────────────
    # Use Ensembl base IDs as primary key (unique; 372 duplicate symbols exist)
    ensembl_ids = adata.var.index.values.astype(str)   # Ensembl base IDs
    symbols     = adata.var[SYMBOL_COL].values.astype(str)

    # ── COVID label → binary int ──────────────────────────────────────────────
    covid_raw = adata.obs[covid_col].astype(str).str.strip()
    print(f"\nCOVID status counts:\n{covid_raw.value_counts().to_string()}")
    y = (covid_raw == covid_pos_label).astype(int).values
    print(f"Encoded: {y.sum()} COVID+ / {(y==0).sum()} COVID-")

    # ── Fetal sex ─────────────────────────────────────────────────────────────
    sex = adata.obs[sex_col].astype(str).str.strip().values
    print(f"\nFetal sex counts:\n{pd.Series(sex).value_counts().to_string()}")

    return adata, X, y, ensembl_ids, symbols, sex


# =============================================================================
# STEP 2 — Variance Filter
# =============================================================================
def variance_filter(adata, ensembl_ids, symbols, top_n=3000):
    """
    Keep top_n most variable genes by variance of log1p(TPM).
    RNA-seq has more low-expressed / zero genes than microarray,
    so we use a slightly larger top_n (3000 vs 2000).
    """
    # gene_vars = np.var(X, axis=0)
    # top_idx   = np.argsort(gene_vars)[::-1][:top_n]
    # print(f"\nVariance filter: {X.shape[1]:,} → {top_n:,} genes retained")
    # print(f"Top 5 most variable genes (Ensembl ID, Symbol, Variance):")
    # for i in range(5):
    #     print(f"  {ensembl_ids[top_idx][i]}, {symbols[top_idx][i]}, {gene_vars[top_idx][i]:.2f}")
    # return X[:, top_idx], ensembl_ids[top_idx], symbols[top_idx] 
    sc.pp.highly_variable_genes(adata, n_top_genes=top_n, flavor='seurat')
    top_idx = np.where(adata.var.highly_variable)[0]
    X_filt = adata.X[:, top_idx] if hasattr(adata.X, "toarray") else adata.X[:, top_idx]
    return X_filt, ensembl_ids[top_idx], symbols[top_idx]

    


# =============================================================================
# STEP 3 — Nested Cross-Validation (unbiased AUC estimate)
# =============================================================================
def nested_cv_auc(X, y, l1_ratios, C_values,
                  ensembl_ids=None, symbols=None, top_n=3000,
                  outer_k=5, inner_k=5, random_state=42):
    """
    Outer loop  → unbiased AUC (the number you report).
    Inner loop  → selects best (C, l1_ratio) per fold.

    Class balance: 39 COVID+ / 26 COVID- → ~60/40 split.
    StratifiedKFold preserves this ratio in every fold.

    Variance filter applied in outer loop (on training data only) before inner search.
    This avoids information leakage from test set during feature selection.

    IMPORTANT: Always report mean CV AUC ± SD, never the in-sample AUC.
    """
    outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True,
                               random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True,
                               random_state=random_state)

    fold_aucs = []
    oof_probs = np.zeros(len(y))
    oof_true  = np.zeros(len(y))

    print(f"\nNested {outer_k}×{inner_k} CV  "
          f"({len(l1_ratios) * len(C_values)} hyperparameter combos per fold)")
    print("-" * 55)

    for fold, (tr, te) in enumerate(outer_cv.split(X, y)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # ── Variance Filter (on training data only, within this fold) ───
        if ensembl_ids is not None:
            # Use scanpy's highly_variable_genes on training fold only (no information leakage from test)
            adata_tr = sc.AnnData(X_tr)
            sc.pp.highly_variable_genes(adata_tr, n_top_genes=top_n, flavor='seurat')
            top_idx = np.where(adata_tr.var.highly_variable)[0]

            # Apply SAME gene indices to both training and test
            X_tr = X_tr[:, top_idx]
            X_te = X_te[:, top_idx]  # Critical: use the same gene subset for test
            
            print(f"  Fold {fold+1}/{outer_k}: Variance filter → {len(top_idx):,} genes")

        # ── Inner grid search ──────────────────────────────
        best_auc, best_C, best_l1 = -1, C_values[0], l1_ratios[0]

        for l1 in l1_ratios:
            for C in C_values:
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf",    LogisticRegression(
                        penalty="elasticnet", solver="saga",
                        C=C, l1_ratio=l1,
                        max_iter=5000, random_state=random_state))
                ])
                inner_aucs = []
                for i_tr, i_te in inner_cv.split(X_tr, y_tr):
                    pipe.fit(X_tr[i_tr], y_tr[i_tr])
                    prob = pipe.predict_proba(X_tr[i_te])[:, 1]
                    if len(np.unique(y_tr[i_te])) > 1:
                        inner_aucs.append(roc_auc_score(y_tr[i_te], prob))
                if inner_aucs and np.mean(inner_aucs) > best_auc:
                    best_auc = np.mean(inner_aucs)
                    best_C, best_l1 = C, l1

        # ── Refit on full training fold with best params ───
        best_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                penalty="elasticnet", solver="saga",
                C=best_C, l1_ratio=best_l1,
                max_iter=5000, random_state=random_state))
        ])
        best_pipe.fit(X_tr, y_tr)
        prob_te = best_pipe.predict_proba(X_te)[:, 1]

        oof_probs[te] = prob_te
        oof_true[te]  = y_te

        auc = roc_auc_score(y_te, prob_te) if len(np.unique(y_te)) > 1 else np.nan
        fold_aucs.append(auc)
        print(f"  Fold {fold+1}/{outer_k}  "
              f"best C={best_C:.4f}  l1_ratio={best_l1}  "
              f"test AUC={auc:.3f}")

    mean_auc = np.nanmean(fold_aucs)
    std_auc  = np.nanstd(fold_aucs)
    print(f"\n  ► Mean CV AUC = {mean_auc:.3f} ± {std_auc:.3f}")
    print("    (Report this number. In-sample AUC will be higher — ignore it.)")
    print("    (Variance filtering applied within outer loop for each fold)")

    return fold_aucs, oof_probs, oof_true


def nested_cv_auc_with_coefficients(X, y, l1_ratios, C_values,
                  ensembl_ids=None, symbols=None, top_n=3000,
                  outer_k=5, inner_k=5, random_state=42):
    """
    Modified nested CV that returns fold-level coefficients and selected genes.
    """
    outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True,
                               random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True,
                               random_state=random_state)

    fold_aucs = []
    oof_probs = np.zeros(len(y))
    oof_true  = np.zeros(len(y))
    fold_coefficients = []  # Store coefficients for each fold
    fold_genes = []  # Store gene names for each fold

    print(f"\nNested {outer_k}×{inner_k} CV (with coefficient tracking)  "
          f"({len(l1_ratios) * len(C_values)} hyperparameter combos per fold)")
    print("-" * 55)

    for fold, (tr, te) in enumerate(outer_cv.split(X, y)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # ── Variance Filter (on training data only, within this fold) ───
        top_idx = None
        if ensembl_ids is not None:
            adata_tr = sc.AnnData(X_tr)
            sc.pp.highly_variable_genes(adata_tr, n_top_genes=top_n, flavor='seurat')
            top_idx = np.where(adata_tr.var.highly_variable)[0]

            X_tr = X_tr[:, top_idx]
            X_te = X_te[:, top_idx]
            
            print(f"  Fold {fold+1}/{outer_k}: Variance filter → {len(top_idx):,} genes")

        # ── Inner grid search ──────────────────────────────
        best_auc, best_C, best_l1 = -1, C_values[0], l1_ratios[0]

        for l1 in l1_ratios:
            for C in C_values:
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf",    LogisticRegression(
                        penalty="elasticnet", solver="saga",
                        C=C, l1_ratio=l1,
                        max_iter=5000, random_state=random_state))
                ])
                inner_aucs = []
                for i_tr, i_te in inner_cv.split(X_tr, y_tr):
                    pipe.fit(X_tr[i_tr], y_tr[i_tr])
                    prob = pipe.predict_proba(X_tr[i_te])[:, 1]
                    if len(np.unique(y_tr[i_te])) > 1:
                        inner_aucs.append(roc_auc_score(y_tr[i_te], prob))
                if inner_aucs and np.mean(inner_aucs) > best_auc:
                    best_auc = np.mean(inner_aucs)
                    best_C, best_l1 = C, l1

        # ── Refit on full training fold with best params ───
        best_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                penalty="elasticnet", solver="saga",
                C=best_C, l1_ratio=best_l1,
                max_iter=5000, random_state=random_state))
        ])
        best_pipe.fit(X_tr, y_tr)
        prob_te = best_pipe.predict_proba(X_te)[:, 1]

        oof_probs[te] = prob_te
        oof_true[te]  = y_te

        auc = roc_auc_score(y_te, prob_te) if len(np.unique(y_te)) > 1 else np.nan
        fold_aucs.append(auc)
        
        # Store fold coefficients
        coef = best_pipe.named_steps['clf'].coef_[0]
        fold_coefficients.append(coef)
        
        # Store gene symbols for this fold
        if top_idx is not None:
            fold_genes.append([symbols[i] for i in top_idx])
        else:
            fold_genes.append(symbols)
        
        print(f"  Fold {fold+1}/{outer_k}  "
              f"best C={best_C:.4f}  l1_ratio={best_l1}  "
              f"test AUC={auc:.3f}")

    mean_auc = np.nanmean(fold_aucs)
    std_auc  = np.nanstd(fold_aucs)
    print(f"\n  ► Mean CV AUC = {mean_auc:.3f} ± {std_auc:.3f}")

    return fold_aucs, oof_probs, oof_true, fold_coefficients, fold_genes


# =============================================================================
# STEP 4 — Stability Selection
# =============================================================================
def stability_selection(X, y, ensembl_ids, symbols,
                        l1_ratio=0.5, C=0.1,
                        n_bootstrap=100,
                        stability_thresh=0.50,
                        random_state=42):
    """
    Fit ElasticNet on n_bootstrap stratified bootstrap resamples.
    A gene is 'stable' if selected in ≥ stability_thresh fraction of resamples.

    Uses Ensembl IDs internally (unique); symbols added for readability.
    """
    n_genes = X.shape[1]
    counts  = np.zeros(n_genes)

    print(f"\nStability selection: {n_bootstrap} bootstrap resamples  "
          f"(l1_ratio={l1_ratio}, C={C})")

    for i in range(n_bootstrap):
        X_boot, y_boot = resample(X, y, stratify=y,
                                  random_state=random_state + i)
        scaler   = StandardScaler()
        X_boot_s = scaler.fit_transform(X_boot)

        clf = LogisticRegression(
            penalty="elasticnet", solver="saga",
            C=C, l1_ratio=l1_ratio,
            max_iter=5000, random_state=random_state)
        clf.fit(X_boot_s, y_boot)
        counts += (clf.coef_[0] != 0).astype(int)

        if (i + 1) % 25 == 0:
            print(f"  Bootstrap {i+1}/{n_bootstrap} done")

    freq = counts / n_bootstrap
    results = pd.DataFrame({
        "ensembl_id": ensembl_ids,
        "symbol":     symbols,
        "frequency":  freq,
        "stable":     freq >= stability_thresh
    }).sort_values("frequency", ascending=False).reset_index(drop=True)

    n_stable = results["stable"].sum()
    print(f"\n  Stable genes (freq ≥ {stability_thresh:.0%}): "
          f"{n_stable} / {n_genes}")

    if n_stable == 0:
        print("  WARNING: No genes passed threshold. "
              "Auto-relaxing to top 10 by frequency.")
        results["stable"] = False
        results.loc[:9, "stable"] = True

    return results


# =============================================================================
# STEP 4b — Build Model Zoo (for multi-model comparison)
# =============================================================================
def build_model_zoo(random_state=42):
    """
    Returns dict of {model_name: sklearn model/pipeline}.
    No StandardScaler wrapping — caller handles scaling per bootstrap.
    """
    models = {
        "ElasticNet LR": LogisticRegression(
            penalty="elasticnet", solver="saga",
            C=0.1, l1_ratio=0.5,
            max_iter=5000, random_state=random_state),
        
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=random_state),
        
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, random_state=random_state),
        
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=2,
            learning_rate=0.05, subsample=0.8,
            random_state=random_state),
        
        "k-NN": KNeighborsClassifier(
            n_neighbors=5, metric="euclidean",
            weights="distance"),
    }
    return models


# =============================================================================
# STEP 4b — Multi-Model Stability Selection
# =============================================================================
def stability_selection_multimodel(X, y, ensembl_ids, symbols,
                                    n_bootstrap=100,
                                    stability_thresh=0.50,
                                    random_state=42):
    """
    Run stability selection for all 5 models independently.
    Returns dict: {model_name: stability_results_df}
    """
    models = build_model_zoo(random_state)
    stability_dict = {}

    print(f"\nMulti-Model Stability Selection: {n_bootstrap} bootstrap resamples")
    print("=" * 70)

    for model_name, model_template in models.items():
        print(f"\n  [{model_name}]")
        n_genes = X.shape[1]
        counts  = np.zeros(n_genes)

        for i in range(n_bootstrap):
            X_boot, y_boot = resample(X, y, stratify=y,
                                      random_state=random_state + i)
            scaler   = StandardScaler()
            X_boot_s = scaler.fit_transform(X_boot)

            # Create fresh model instance
            if model_name == "ElasticNet LR":
                clf = LogisticRegression(
                    penalty="elasticnet", solver="saga",
                    C=0.1, l1_ratio=0.5,
                    max_iter=5000, random_state=random_state)
            elif model_name == "Random Forest":
                clf = RandomForestClassifier(
                    n_estimators=500, max_features="sqrt",
                    min_samples_leaf=2, random_state=random_state)
            elif model_name == "SVM (RBF)":
                clf = SVC(kernel="rbf", C=1.0, gamma="scale",
                         probability=True, random_state=random_state)
            elif model_name == "Gradient Boosting":
                clf = GradientBoostingClassifier(
                    n_estimators=200, max_depth=2,
                    learning_rate=0.05, subsample=0.8,
                    random_state=random_state)
            elif model_name == "k-NN":
                clf = KNeighborsClassifier(
                    n_neighbors=5, metric="euclidean",
                    weights="distance")

            clf.fit(X_boot_s, y_boot)

            # Extract feature importance/coefficients
            if hasattr(clf, 'coef_'):  # Linear models
                counts += (clf.coef_[0] != 0).astype(int)
            elif hasattr(clf, 'feature_importances_'):  # Tree-based
                counts += (clf.feature_importances_ > 0).astype(int)
            elif model_name == "k-NN":  # k-NN doesn't have feature importance
                counts += 1  # All features "used" equally

            if (i + 1) % 25 == 0:
                print(f"    Bootstrap {i+1}/{n_bootstrap} done")

        freq = counts / n_bootstrap
        results = pd.DataFrame({
            "ensembl_id": ensembl_ids,
            "symbol":     symbols,
            "frequency":  freq,
            "stable":     freq >= stability_thresh
        }).sort_values("frequency", ascending=False).reset_index(drop=True)

        n_stable = results["stable"].sum()
        print(f"    ✓ Stable genes (freq ≥ {stability_thresh:.0%}): {n_stable} / {n_genes}")

        if n_stable == 0:
            print(f"    WARNING: No genes passed threshold. Auto-relaxing to top 10.")
            results["stable"] = False
            results.loc[:9, "stable"] = True

        stability_dict[model_name] = results

    return stability_dict


# =============================================================================
# STEP 5 — Final Model on Stable Panel
# =============================================================================
def fit_final_model(X, y, ensembl_ids, symbols, stability_df,
                    l1_ratio=0.5, C=0.1, random_state=42):
    """
    Fit ElasticNet on stable gene panel only.
    Returns fitted pipeline and coefficient DataFrame.
    """
    stable_set   = set(stability_df.loc[stability_df["stable"], "ensembl_id"])
    stable_mask  = np.array([g in stable_set for g in ensembl_ids])

    X_panel      = X[:, stable_mask]
    panel_ensembl = ensembl_ids[stable_mask]
    panel_symbols = symbols[stable_mask]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            penalty="elasticnet", solver="saga",
            C=C, l1_ratio=l1_ratio,
            max_iter=5000, random_state=random_state))
    ])
    pipe.fit(X_panel, y)
    coefs = pipe.named_steps["clf"].coef_[0]

    panel_df = pd.DataFrame({
        "ensembl_id":  panel_ensembl,
        "symbol":      panel_symbols,
        "coefficient": coefs
    }).sort_values("coefficient", key=abs, ascending=False).reset_index(drop=True)

    print(f"\nFinal panel ({len(panel_df)} genes):")
    print(panel_df.to_string(index=False))

    return pipe, panel_df, stable_mask

def evaluate_all_panels(X, y, ensembl_ids, stability_dict, random_state=42):
    """
    Takes the 'stable' genes from each model and tests them using a standard
    ElasticNet Logistic Regression to see which panel is most 'portable'.
    """
    panel_performance = []
    
    print("\n" + "=" * 60)
    print("EVALUATING PANEL PORTABILITY (Final Model = ElasticNet)")
    print("=" * 60)

    for model_name, stability_df in stability_dict.items():
        stable_genes = stability_df.loc[stability_df["stable"], "ensembl_id"].values
        
        if len(stable_genes) == 0:
            print(f"  [{model_name}] No stable genes found. Skipping.")
            continue
            
        # Create mask for the specific genes picked by THIS model
        mask = np.isin(ensembl_ids, stable_genes)
        X_sub = X[:, mask]
        
        # Standard Nested CV to evaluate this specific gene set
        # We use a fixed ElasticNet to see how well these features perform
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = []
        
        for tr, te in cv.split(X_sub, y):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(penalty="elasticnet", solver="saga", 
                                           C=0.1, l1_ratio=0.5, max_iter=5000))
            ])
            pipe.fit(X_sub[tr], y[tr])
            probs = pipe.predict_proba(X_sub[te])[:, 1]
            scores.append(roc_auc_score(y[te], probs))
            
        mean_s = np.mean(scores)
        std_s = np.std(scores)
        panel_performance.append({
            "Source Model": model_name,
            "Panel Size": len(stable_genes),
            "Mean AUC": mean_s,
            "Std AUC": std_s
        })
        print(f"  [{model_name} Panel] Size: {len(stable_genes)} | AUC: {mean_s:.3f} ± {std_s:.3f}")
        
    return pd.DataFrame(panel_performance)


# =============================================================================
# PLOTTING
# =============================================================================
def plot_stability_barplot(stability_df, top_n=30, outpath=None):
    df = stability_df.head(top_n).copy().sort_values("frequency")
    # Use symbol for display; fall back to ensembl_id if symbol is nan
    df["label"] = df["symbol"].where(df["symbol"] != "nan", df["ensembl_id"])

    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.28)))
    colors = [COL_STABLE if s else COL_UNSTABLE for s in df["stable"]]
    bars = ax.barh(df["label"], df["frequency"],
                   color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(STABILITY_THRESH, color="black", linestyle="--",
               linewidth=1.2, label=f"Threshold ({STABILITY_THRESH:.0%})")
    ax.set_xlabel("Bootstrap selection frequency", fontsize=11)
    ax.set_title(f"Stability Selection — Top {top_n} Genes\n"
                 f"(COVID+ vs COVID−, log1p TPM, ElasticNet)",
                 fontsize=11, pad=10)
    ax.set_xlim(0, 1.05)
    for bar, freq in zip(bars, df["frequency"]):
        ax.text(freq + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{freq:.2f}", va="center", fontsize=7.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=COL_STABLE,   label=f"Stable (≥{STABILITY_THRESH:.0%})"),
        Patch(facecolor=COL_UNSTABLE, label=f"Unstable (<{STABILITY_THRESH:.0%})")
    ], fontsize=9, loc="lower right")
    sns.despine()
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"  Saved → {outpath}")
    plt.close()


def plot_coefficient_forest(panel_df, outpath=None):
    df = panel_df.sort_values("coefficient").copy()
    df["label"] = df["symbol"].where(df["symbol"] != "nan", df["ensembl_id"])
    colors = [COL_COVID_POS if c > 0 else COL_COVID_NEG for c in df["coefficient"]]

    fig, ax = plt.subplots(figsize=(6, max(3, len(df) * 0.38)))
    ax.barh(df["label"], df["coefficient"],
            color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("ElasticNet coefficient", fontsize=11)
    ax.set_title("Stable Gene Panel — Coefficients\n"
                 "(positive = higher in COVID+)", fontsize=11, pad=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=COL_COVID_POS, label="Higher in COVID+ (+)"),
        Patch(facecolor=COL_COVID_NEG, label="Lower in COVID+ (−)")
    ], fontsize=9)
    sns.despine()
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"  Saved → {outpath}")
    plt.close()


def plot_roc_pr(oof_true, oof_probs, fold_aucs, outpath=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ROC
    fpr, tpr, _ = roc_curve(oof_true, oof_probs)
    mean_auc = np.nanmean(fold_aucs)
    std_auc  = np.nanstd(fold_aucs)
    axes[0].plot(fpr, tpr, color=COL_COVID_POS, lw=2,
                 label=f"OOF ROC  (mean CV AUC = {mean_auc:.3f} ± {std_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    axes[0].fill_between(fpr, tpr, alpha=0.12, color=COL_COVID_POS)
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].set_title("ROC Curve (OOF)", fontsize=12)
    axes[0].legend(fontsize=9)

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(oof_true, oof_probs)
    ap       = average_precision_score(oof_true, oof_probs)
    baseline = oof_true.mean()
    axes[1].plot(rec, prec, color=COL_COVID_NEG, lw=2,
                 label=f"OOF PR  (AP = {ap:.3f})")
    axes[1].axhline(baseline, color="gray", linestyle="--", lw=1,
                    label=f"Baseline (prevalence = {baseline:.2f})")
    axes[1].fill_between(rec, prec, alpha=0.12, color=COL_COVID_NEG)
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].set_title("Precision-Recall Curve (OOF)", fontsize=12)
    axes[1].legend(fontsize=9)

    for ax in axes:
        sns.despine(ax=ax)
    plt.suptitle("Discovery cohort only — no external validation performed",
                 fontsize=9, color="gray", style="italic", y=1.01)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"  Saved → {outpath}")
    plt.close()


def plot_sex_stratified_check(oof_probs, oof_true, sex, outpath=None):
    """
    Sanity check: compute AUC separately for Female and Male fetal sex subgroups.
    The source paper (Pinatel et al. 2026) found a strong sex-specific response
    (43 DEGs in XX COVID+ vs 1 DEG in XY COVID+).
    If the classifier captures sex-specific signal, AUC will differ by subgroup.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    sex_groups = {"Female": COL_FEMALE, "Male": COL_MALE}
    for ax, (sex_label, color) in zip(axes, sex_groups.items()):
        mask = sex == sex_label
        if mask.sum() < 5 or len(np.unique(oof_true[mask])) < 2:
            ax.text(0.5, 0.5, f"Insufficient data\n({mask.sum()} samples)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Fetal sex: {sex_label}", fontsize=11)
            continue

        fpr, tpr, _ = roc_curve(oof_true[mask], oof_probs[mask])
        auc = roc_auc_score(oof_true[mask], oof_probs[mask])
        n_pos = oof_true[mask].sum()
        n_neg = (oof_true[mask] == 0).sum()

        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"AUC = {auc:.3f}\n(n={mask.sum()}: {n_pos:.0f} COVID+, {n_neg:.0f} COVID−)")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.fill_between(fpr, tpr, alpha=0.12, color=color)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(f"Fetal sex: {sex_label}", fontsize=11)
        ax.legend(fontsize=9)
        sns.despine(ax=ax)

    plt.suptitle("Sex-Stratified AUC — Sanity Check\n"
                 "(Pinatel et al. 2026 found sex-specific DEGs: 43 in XX, 1 in XY)",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"  Saved → {outpath}")
    plt.close()


# =============================================================================
# MULTI-MODEL COMPARISON PLOTS
# =============================================================================
def plot_model_panel_sizes(stability_dict, outpath=None):
    """
    Barplot: number of stable genes per model.
    """
    sizes = {name: df["stable"].sum() for name, df in stability_dict.items()}
    df = pd.DataFrame({
        "model": list(sizes.keys()),
        "n_genes": list(sizes.values())
    }).sort_values("n_genes", ascending=True)

    colors = [MODEL_COLORS.get(m, "#888888") for m in df["model"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(df["model"], df["n_genes"],
                   color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of stable genes", fontsize=11)
    ax.set_title("Stable Gene Panel Sizes — All Models\n"
                 f"(Stability threshold: {STABILITY_THRESH:.0%})",
                 fontsize=11, pad=10)
    for bar, n in zip(bars, df["n_genes"]):
        ax.text(n + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{int(n)}", va="center", fontsize=10)
    sns.despine()
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"  Saved → {outpath}")
    plt.close()


def plot_model_agreement_heatmap(stability_dict, outpath=None):
    """
    Heatmap: for each model, mark which genes are stable.
    Show agreement across models.
    """
    # Get union of all stable genes
    all_stable_genes = set()
    for df in stability_dict.values():
        all_stable_genes.update(df.loc[df["stable"], "ensembl_id"])
    
    # Rank by total frequency across models
    gene_freqs = {}
    for df in stability_dict.values():
        for _, row in df.iterrows():
            eid = row["ensembl_id"]  # Use symbol for display; could also use ensembl_id
            if eid in all_stable_genes:
                gene_freqs[eid] = gene_freqs.get(eid, 0) + row["frequency"]
    
    top_genes = sorted(gene_freqs.items(), key=lambda x: x[1], reverse=True)[:15]
    top_gene_ids = [g[0] for g in top_genes]

    # Build matrix: rows = genes, cols = models
    matrix_data = []
    for gene_id in top_gene_ids:
        row = []
        for model_name in stability_dict.keys():
            df = stability_dict[model_name]
            freq = df[df["ensembl_id"] == gene_id]["frequency"].values
            row.append(freq[0] if len(freq) > 0 else 0)
        matrix_data.append(row)

    # use symbols for display; fall back to ensembl_id if symbol is nan
    gene_labels = []
    for gene_id in top_gene_ids:
        symbol = None
        for df in stability_dict.values():
            sym = df[df["ensembl_id"] == gene_id]["symbol"].values
            if len(sym) > 0 and sym[0] != "nan":
                symbol = sym[0]
                break
        gene_labels.append(symbol if symbol else gene_id)

    df_heatmap = pd.DataFrame(
        matrix_data,
        index=gene_labels,
        columns=list(stability_dict.keys())
    )
    
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.heatmap(df_heatmap, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=1, linewidths=0.5,
                linecolor="white", ax=ax,
                cbar_kws={"label": "Selection frequency"})
    ax.set_title("Gene Selection Agreement Across Models\n"
                 "(Top 15 genes by total stability frequency)",
                 fontsize=11, pad=10)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Gene", fontsize=11)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"  Saved → {outpath}")
    plt.close()

def plot_panel_comparison(performance_df, outpath=None):
    plt.figure(figsize=(8, 5))
    df = performance_df.sort_values("Mean AUC", ascending=False)
    
    colors = [MODEL_COLORS.get(m, "#888888") for m in df["Source Model"]]
    
    ax = sns.barplot(x="Mean AUC", y="Source Model", data=df, palette=colors)
    plt.errorbar(x=df["Mean AUC"], y=range(len(df)), xerr=df["Std AUC"], 
                 fmt='none', c='black', capsize=3)
    
    plt.title("Predictive Power of Model-Specific Gene Panels\n(All tested via final ElasticNet)", fontsize=12)
    plt.xlabel("Mean Cross-Validation AUC", fontsize=11)
    plt.xlim(0.5, 1.0) # Focus on the performance range
    sns.despine()
    
    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# PLOTTING FUNCTIONS FOR NEW FIGURES
# =============================================================================

def plot_gene_contributions_heatmap(fold_coefficients, fold_genes, outpath, top_n=30):
    """
    Heatmap of Gene Contributions Across Folds.
    
    Visualizes the variability of gene contributions across CV folds.
    Shows top genes by mean absolute contribution.
    """
    # Find common genes across all folds (or use top genes)
    all_genes_set = set()
    for genes in fold_genes:
        all_genes_set.update(genes)
    all_genes_list = sorted(list(all_genes_set))
    
    # Create matrix: folds x genes
    data_list = []
    for fold_idx, (coef, genes) in enumerate(zip(fold_coefficients, fold_genes)):
        fold_data = {}
        for gene, c in zip(genes, coef):
            fold_data[gene] = c
        data_list.append(fold_data)
    
    df = pd.DataFrame(data_list).fillna(0)
    
    # Select top genes by mean absolute contribution
    mean_abs_contrib = df.abs().mean(axis=0).sort_values(ascending=False)
    top_genes = mean_abs_contrib.head(top_n).index.tolist()
    df_top = df[top_genes]
    
    # Create heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_top.T, cmap='RdBu_r', center=0, cbar_kws={'label': 'Coefficient'},
                xticklabels=[f"Fold {i+1}" for i in range(len(fold_coefficients))],
                yticklabels=top_genes)
    plt.title(f"Top {top_n} Gene Contributions Across CV Folds")
    plt.xlabel("CV Fold")
    plt.ylabel("Gene")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  ✓ Gene contributions heatmap saved: {outpath}")


def plot_sex_stratified_performance(oof_probs, oof_true, sex, outpath):
    """
    Sex-Stratified Performance: AUC and Accuracy by sex.
    
    Shows model performance separately for male and female samples.
    """
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    sex_unique = np.unique(sex)
    aucs = []
    accs = []
    sex_labels = []
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for s in sex_unique:
        mask = sex == s
        if np.sum(mask) > 0 and len(np.unique(oof_true[mask])) > 1:
            auc = roc_auc_score(oof_true[mask], oof_probs[mask])
            acc = accuracy_score(oof_true[mask], (oof_probs[mask] > 0.5).astype(int))
            aucs.append(auc)
            accs.append(acc)
            sex_labels.append(s)
    
    # AUC by sex
    colors = [COL_FEMALE if s.lower() == "female" else COL_MALE for s in sex_labels]
    ax1.bar(sex_labels, aucs, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel("AUC")
    ax1.set_title("Nested CV AUC by Fetal Sex")
    ax1.set_ylim([0, 1])
    for i, v in enumerate(aucs):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    # Accuracy by sex
    ax2.bar(sex_labels, accs, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy by Fetal Sex")
    ax2.set_ylim([0, 1])
    for i, v in enumerate(accs):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  ✓ Sex-stratified performance saved: {outpath}")


def plot_gene_expression_boxplots(X, y, ensembl_ids, symbols, adata, top_n_genes=10, outpath=None):
    """
    Gene Expression Boxplots: Expression levels of top genes by COVID status.
    
    Shows distribution of expression for top genes across COVID+ and COVID- samples.
    """
    from sklearn.preprocessing import StandardScaler
    
    # Get top genes based on variance in scaled data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    gene_variances = np.var(X_scaled, axis=0)
    top_idx = np.argsort(gene_variances)[-top_n_genes:][::-1]
    
    top_genes = [symbols[i] for i in top_idx]
    X_top = X[:, top_idx]
    
    # Create boxplots
    n_genes = len(top_genes)
    n_cols = 5
    n_rows = (n_genes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, (gene_idx, gene_name) in enumerate(zip(top_idx, top_genes)):
        ax = axes[idx]
        data_by_class = [X_top[y == 0, idx], X_top[y == 1, idx]]
        labels = ["COVID-", "COVID+"]
        colors = [COL_COVID_NEG, COL_COVID_POS]
        
        bp = ax.boxplot(data_by_class, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(gene_name, fontweight='bold')
        ax.set_ylabel("Expression (log1p TPM)")
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(top_genes), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"Expression Patterns of Top {top_n_genes} Genes by COVID Status", 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gene expression boxplots saved: {outpath}")


def plot_confusion_matrix(oof_true, oof_probs, outpath):
    """
    Confusion Matrix: Summarize classification performance.
    
    Shows TP, TN, FP, FN for the final model.
    """
    from sklearn.metrics import confusion_matrix
    
    # Use probability threshold of 0.5
    predictions = (oof_probs > 0.5).astype(int)
    cm = confusion_matrix(oof_true, predictions)
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['COVID-', 'COVID+'],
                yticklabels=['COVID-', 'COVID+'],
                cbar_kws={'label': 'Count'})
    plt.title("Confusion Matrix (Out-of-Fold Predictions)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    # Add metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    metrics_text = f"Sensitivity: {sensitivity:.3f}\nSpecificity: {specificity:.3f}\nPPV: {ppv:.3f}\nNPV: {npv:.3f}"
    plt.text(2.5, -0.5, metrics_text, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confusion matrix saved: {outpath}")


def plot_model_feature_importance_comparison(stability_dict, top_n=15, outpath=None):
    """
    Feature Importance Comparison Across Models.
    
    Grouped barplot comparing gene importance across different models.
    """
    # Get top genes from ElasticNet
    elasticnet_df = stability_dict["ElasticNet LR"]
    top_genes_en = elasticnet_df.nlargest(top_n, "frequency")["symbol"].tolist()
    
    # Prepare data for grouped barplot
    plot_data = []
    for gene in top_genes_en:
        for model_name, stability_df in stability_dict.items():
            freq = stability_df[stability_df["symbol"] == gene]["frequency"].values
            freq = freq[0] if len(freq) > 0 else 0
            plot_data.append({"Gene": gene, "Model": model_name, "Frequency": freq})
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create grouped barplot
    fig, ax = plt.subplots(figsize=(14, 6))
    genes = plot_df["Gene"].unique()
    x = np.arange(len(genes))
    width = 0.15
    
    models = list(stability_dict.keys())
    for idx, model in enumerate(models):
        model_data = plot_df[plot_df["Model"] == model]
        freqs = [model_data[model_data["Gene"] == g]["Frequency"].values[0] 
                 if len(model_data[model_data["Gene"] == g]) > 0 else 0 for g in genes]
        color = MODEL_COLORS.get(model, "#888888")
        ax.bar(x + idx*width, freqs, width, label=model, color=color, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel("Gene", fontweight='bold')
    ax.set_ylabel("Selection Frequency", fontweight='bold')
    ax.set_title(f"Feature Importance Comparison - Top {top_n} Genes from ElasticNet")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(genes, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Model feature importance comparison saved: {outpath}")


def plot_final_model_sex_stratified_roc(X, y, sex, ensembl_ids, stable_mask, l1_ratio=0.5, C=0.1, outpath=None):
    """
    Sex-Stratified ROC Curves for ElasticNet Final Model.
    
    Trains ElasticNet on the stable gene panel and evaluates separately for each sex.
    Shows ROC curves and AUC for Male and Female subgroups.
    """
    X_panel = X[:, stable_mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sex_groups = {"Female": (COL_FEMALE, 0), "Male": (COL_MALE, 1)}
    
    for ax, (sex_label, (color, ax_idx)) in zip(axes, sex_groups.items()):
        mask = sex == sex_label
        
        if mask.sum() < 5 or len(np.unique(y[mask])) < 2:
            ax.text(0.5, 0.5, f"Insufficient data\n({mask.sum()} samples)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title(f"Fetal sex: {sex_label}", fontsize=11)
            continue
        
        X_sex = X_panel[mask]
        y_sex = y[mask]
        
        # Use nested CV to get OOF predictions
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        probs_sex = np.zeros(len(y_sex))
        
        for tr, te in cv.split(X_sex, y_sex):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="elasticnet", solver="saga",
                    C=C, l1_ratio=l1_ratio,
                    max_iter=5000, random_state=42))
            ])
            pipe.fit(X_sex[tr], y_sex[tr])
            probs_sex[te] = pipe.predict_proba(X_sex[te])[:, 1]
        
        # Plot ROC
        fpr, tpr, _ = roc_curve(y_sex, probs_sex)
        auc = roc_auc_score(y_sex, probs_sex)
        n_pos = y_sex.sum()
        n_neg = (y_sex == 0).sum()
        
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"AUC = {auc:.3f}\n(n={mask.sum()}: {n_pos:.0f} COVID+, {n_neg:.0f} COVID−)")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.fill_between(fpr, tpr, alpha=0.12, color=color)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(f"Fetal sex: {sex_label}", fontsize=11)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(alpha=0.3)
        sns.despine(ax=ax)
    
    plt.suptitle("Sex-Stratified ROC Curves — ElasticNet Final Model (Stable Gene Panel)",
                 fontsize=12, fontweight='bold', y=1.00)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f"  ✓ Sex-stratified ROC curves saved: {outpath}")
    plt.close()


def plot_panel_roc_comparison(X, y, ensembl_ids, stability_dict, l1_ratio=0.5, C=0.1, outpath=None):
    """
    ROC Curves for ElasticNet Models Trained on Different Gene Panels.
    
    For each gene panel (ElasticNet, Random Forest, SVM, Gradient Boosting, k-NN),
    train an ElasticNet model and plot the ROC curve to compare panel performance.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    
    roc_data = []
    
    for model_name, stability_df in stability_dict.items():
        stable_genes = stability_df.loc[stability_df["stable"], "ensembl_id"].values
        
        if len(stable_genes) == 0:
            print(f"  [{model_name}] No stable genes. Skipping.")
            continue
        
        # Create mask for these genes
        mask = np.isin(ensembl_ids, stable_genes)
        X_sub = X[:, mask]
        
        # Nested CV to get OOF predictions
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        probs = np.zeros(len(y))
        
        for tr, te in cv.split(X_sub, y):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="elasticnet", solver="saga",
                    C=C, l1_ratio=l1_ratio,
                    max_iter=5000, random_state=42))
            ])
            pipe.fit(X_sub[tr], y[tr])
            probs[te] = pipe.predict_proba(X_sub[te])[:, 1]
        
        # Calculate ROC
        fpr, tpr, _ = roc_curve(y, probs)
        auc = roc_auc_score(y, probs)
        
        color = MODEL_COLORS.get(model_name, "#888888")
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{model_name} (AUC={auc:.3f}, n={len(stable_genes)} genes)")
        
        roc_data.append({
            "model": model_name,
            "auc": auc,
            "n_genes": len(stable_genes)
        })
    
    # Add diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random classifier")
    
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — ElasticNet Models Trained on Different Gene Panels",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    sns.despine(ax=ax)
    
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f"  ✓ Panel ROC comparison saved: {outpath}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
# if __name__ == "__main__":

#     print("=" * 60)
#     print("STEP 1: Load Data")
#     print("=" * 60)
#     X, y, ensembl_ids, symbols, sex = load_data(
#         ANNDATA_PATH, COVID_COL, SEX_COL, COVID_POS_LABEL
#     )

#     print("\n" + "=" * 60)
#     print("STEP 2: Variance Filter — Global (for later use)")
#     print("="*60)
#     X_filt, ensembl_filt, symbols_filt = variance_filter(
#         X, ensembl_ids, symbols, top_n=TOP_N_GENES
#     )

#     print("\n" + "="*60)
#     print("STEP 3: Nested Cross-Validation (with fold-wise variance filter)")
#     print("="*60)
#     fold_aucs, oof_probs, oof_true = nested_cv_auc(
#         X, y, # filtering applied within outer loop
#         l1_ratios=L1_RATIOS,
#         C_values=C_VALUES,
#         ensembl_ids=ensembl_ids,
#         symbols=symbols,
#         top_n=TOP_N_GENES,
#         outer_k=5, inner_k=5,
#         random_state=RANDOM_STATE
#     )

#     print("\n" + "=" * 60)
#     print("STEP 4: Multi-Model Stability Selection")
#     print("=" * 60)
#     stability_dict = stability_selection_multimodel(
#         X, y, ensembl_filt, symbols_filt, # previously filtered gene set
#         n_bootstrap=N_BOOTSTRAP,
#         stability_thresh=STABILITY_THRESH,
#         random_state=RANDOM_STATE
#     )

#     # Save all model panels
#     print("\nSaving stable panels for all models...")
#     for model_name, stability_df in stability_dict.items():
#         stability_df.to_csv(
#             f"{OUT_DIR}/{model_name.replace(' ', '_').lower()}_stability_all_genes.csv",
#             index=False
#         )
#         stable_genes = stability_df[stability_df["stable"]]
#         stable_genes.to_csv(
#             f"{OUT_DIR}/{model_name.replace(' ', '_').lower()}_stable_gene_panel.csv",
#             index=False
#         )
#     print(f"  ✓ All panels saved to {OUT_DIR}/")

#     print("\n" + "=" * 60)
#     print("STEP 5: Final Model (using ElasticNet panel)")
#     print("=" * 60)
#     # Use the ElasticNet stability results from the multi-model comparison
#     elasticnet_stability_df = stability_dict["ElasticNet LR"]
#     final_model, panel_df, stable_mask = fit_final_model(
#         X_filt, y, ensembl_filt, symbols_filt, elasticnet_stability_df, # use the previously variance-filtered gene set
#         l1_ratio=0.5, C=0.1,
#         random_state=RANDOM_STATE
#     )

#     # ── Save CSVs ──────────────────────────────────────────────────────────────
#     panel_df.to_csv(f"{OUT_DIR}/stable_gene_panel.csv", index=False)
#     stability_df.to_csv(f"{OUT_DIR}/stability_all_genes.csv", index=False)
#     print(f"\nCSVs saved to {OUT_DIR}/")

#     # ── Figures ────────────────────────────────────────────────────────────────
#     print("\n" + "=" * 60)
#     print("STEP 6: Generating Figures")
#     print("=" * 60)
#     plot_stability_barplot(
#         stability_df, top_n=30,
#         outpath=f"{OUT_DIR}/fig1_stability_barplot.png"
#     )
#     plot_coefficient_forest(
#         panel_df,
#         outpath=f"{OUT_DIR}/fig2_coefficient_forest.png"
#     )
#     plot_roc_pr(
#         oof_true, oof_probs, fold_aucs,
#         outpath=f"{OUT_DIR}/fig3_roc_pr_curves.png"
#     )
#     plot_sex_stratified_check(
#         oof_probs, oof_true, sex,
#         outpath=f"{OUT_DIR}/fig4_sex_stratified_check.png"
#     )

#     # Multi-model comparison plots
#     print("\n  Generating multi-model comparison plots...")
#     plot_model_panel_sizes(
#         stability_dict,
#         outpath=f"{OUT_DIR}/fig5_model_panel_sizes.png"
#     )
#     plot_model_agreement_heatmap(
#         stability_dict,
#         outpath=f"{OUT_DIR}/fig6_model_agreement_heatmap.png"
#     )

#     print("\n" + "=" * 60)
#     print("PIPELINE COMPLETE")
#     print("=" * 60)
#     print(f"  Mean CV AUC : {np.nanmean(fold_aucs):.3f} ± {np.nanstd(fold_aucs):.3f}")
#     print(f"  Stable genes: {panel_df.shape[0]}")
#     print(f"  Outputs in  : {OUT_DIR}/")
#     print("\n  REMINDER: AUC is a discovery-cohort estimate only.")
#     print("  No external validation was performed.")
#     print("  Source paper: Pinatel et al. 2026 (GSE318446)")

if __name__ == "__main__":

    print("=" * 60)
    print("STEP 1: Load Data")
    print("=" * 60)
    adata, X, y, ensembl_ids, symbols, sex = load_data(
        ANNDATA_PATH, COVID_COL, SEX_COL, COVID_POS_LABEL
    )

    print("\n" + "=" * 60)
    print("STEP 2: Variance Filter — Global (for later use)")
    print("="*60)
    X_filt, ensembl_filt, symbols_filt = variance_filter(
        adata, ensembl_ids, symbols, top_n=TOP_N_GENES
    )

    print("\n" + "="*60)
    print("STEP 3: Nested Cross-Validation (Unbiased Baseline)")
    print("="*60)
    # This gives us the baseline AUC using the standard pipeline
    fold_aucs, oof_probs, oof_true, fold_coefficients, fold_genes = nested_cv_auc_with_coefficients(
        X, y, 
        l1_ratios=L1_RATIOS,
        C_values=C_VALUES,
        ensembl_ids=ensembl_ids,
        symbols=symbols,
        top_n=TOP_N_GENES,
        outer_k=5, inner_k=5,
        random_state=RANDOM_STATE
    )

    print("\n" + "=" * 60)
    print("STEP 4: Multi-Model Stability Selection")
    print("=" * 60)
    # Identify important genes for all 5 algorithms
    stability_dict = stability_selection_multimodel(
        X, y, ensembl_ids, symbols, # previously filtered gene set X_filt, ensembl_filt, symbols_filt,
        n_bootstrap=N_BOOTSTRAP,
        stability_thresh=STABILITY_THRESH,
        random_state=RANDOM_STATE
    )

# Save all model panels
    print("\nSaving stable panels for all models...")
    for model_name, stability_df in stability_dict.items():
        stability_df.to_csv(
            f"{OUT_DIR}/{model_name.replace(' ', '_').lower()}_stability_all_genes.csv",
            index=False
        )
        stable_genes = stability_df[stability_df["stable"]]
        stable_genes.to_csv(
            f"{OUT_DIR}/{model_name.replace(' ', '_').lower()}_stable_gene_panel.csv",
            index=False
        )
    print(f"  ✓ All panels saved to {OUT_DIR}/")

    print("\n" + "=" * 60)
    print("STEP 5: Cross-Panel Portability Evaluation")
    print("=" * 60)
    # NEW: Test the gene panels from different models against each other
    performance_df = evaluate_all_panels(X, y, ensembl_ids, stability_dict) # previously filtered gene set
    performance_df.to_csv(f"{OUT_DIR}/panel_portability_comparison.csv", index=False)

    print("\n" + "=" * 60)
    print("STEP 6: Final Model (Reference ElasticNet Panel)")
    print("=" * 60)
    elasticnet_stability_df = stability_dict["ElasticNet LR"]
    final_model, panel_df, stable_mask = fit_final_model(
        X, y, ensembl_ids, symbols, elasticnet_stability_df, # use the previously variance-filtered gene set
        l1_ratio=0.5, C=0.1,
        random_state=RANDOM_STATE
    )

    # ── Save Results ──────────────────────────────────────────────────────────────
    panel_df.to_csv(f"{OUT_DIR}/stable_gene_panel.csv", index=False)
    print(f"\nResults saved to {OUT_DIR}/")

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Generating Figures")
    print("=" * 60)
    
    # Original Figures
    plot_stability_barplot(elasticnet_stability_df, top_n=30, outpath=f"{OUT_DIR}/fig1_stability_barplot.png")
    plot_coefficient_forest(panel_df, outpath=f"{OUT_DIR}/fig2_coefficient_forest.png")
    plot_roc_pr(oof_true, oof_probs, fold_aucs, outpath=f"{OUT_DIR}/fig3_roc_pr_curves.png")
    plot_sex_stratified_check(oof_probs, oof_true, sex, outpath=f"{OUT_DIR}/fig4_sex_stratified_check.png")

    # Multi-model Comparison Figures
    plot_model_panel_sizes(stability_dict, outpath=f"{OUT_DIR}/fig5_model_panel_sizes.png")
    plot_model_agreement_heatmap(stability_dict, outpath=f"{OUT_DIR}/fig6_model_agreement_heatmap.png")
    
    # Panel Performance Figure
    plot_panel_comparison(performance_df, outpath=f"{OUT_DIR}/fig7_panel_performance.png")
    
    # Gene Contributions and Additional Analyses
    print("\n  Generating additional analysis figures...")
    plot_gene_contributions_heatmap(fold_coefficients, fold_genes, 
                                     outpath=f"{OUT_DIR}/fig8_gene_contributions_heatmap.png", top_n=30)
    plot_sex_stratified_performance(oof_probs, oof_true, sex, 
                                     outpath=f"{OUT_DIR}/fig9_sex_stratified_performance.png")
    plot_gene_expression_boxplots(X, y, ensembl_ids, symbols, adata, top_n_genes=10,
                                   outpath=f"{OUT_DIR}/fig10_gene_expression_boxplots.png")
    plot_confusion_matrix(oof_true, oof_probs, 
                          outpath=f"{OUT_DIR}/fig11_confusion_matrix.png")
    plot_model_feature_importance_comparison(stability_dict, top_n=15, 
                                              outpath=f"{OUT_DIR}/fig12_model_feature_importance.png")
    
    # Sex-stratified ROC for final ElasticNet model
    plot_final_model_sex_stratified_roc(X, y, sex, ensembl_ids, stable_mask,
                                         l1_ratio=0.5, C=0.1,
                                         outpath=f"{OUT_DIR}/fig13_final_model_sex_stratified_roc.png")
    
    # ROC curves comparing different gene panels
    plot_panel_roc_comparison(X, y, ensembl_ids, stability_dict, 
                              l1_ratio=0.5, C=0.1,
                              outpath=f"{OUT_DIR}/fig14_panel_roc_comparison.png")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
