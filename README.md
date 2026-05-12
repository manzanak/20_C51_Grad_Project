# COVID-19 Classification from Placental RNA-seq

A comprehensive machine learning pipeline for COVID-19 status classification using bulk RNA-seq gene expression data from placental tissue.

## Features

- **Nested Cross-Validation**: Unbiased AUC estimation (5×5 folds)
- **Multi-Model Stability Selection**: 5 algorithms (ElasticNet, Random Forest, SVM, Gradient Boosting, k-NN)
- **Robust Feature Selection**: 100 bootstrap resamples per model
- **Sex-Stratified Analysis**: Separate evaluation for Male/Female samples
- **Comprehensive Visualization**: 14 publication-ready figures

## Quick Start

```bash
# Install dependencies
conda create -n covid_placenta python=3.10
conda activate covid_placenta
pip install numpy pandas scikit-learn scanpy matplotlib seaborn scipy

# Run pipeline
python covid_multimodel_feature_selection_pipeline.py
