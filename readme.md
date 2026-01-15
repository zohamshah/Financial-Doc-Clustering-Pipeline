# Financial Document Clustering (Preprocessing × Representation × Clustering Evaluation)

This notebook implements an end-to-end pipeline that converts finance/operational records into text “documents”, applies multiple preprocessing variants, generates alternative text representations (TF-IDF+LSA vs embeddings), performs clustering, and evaluates clustering quality using internal validation indices. It also produces comparison plots and exports a results table.

## Repository Contents

- `Financial_document_clustering.ipynb`  
  Main notebook containing:
  - data loading
  - record-to-document construction
  - preprocessing variants (P0–P3)
  - representations (TF-IDF+LSA and embeddings)
  - clustering (e.g., K-Means)
  - evaluation metrics (Silhouette, Davies–Bouldin, Calinski–Harabasz)
  - plots + CSV export

## High-Level Pipeline

1. Load operational/financial records (CSV) into a dataframe
2. Build a document per record by concatenating selected fields (record-to-document construction)
3. Apply preprocessing variants (P0–P3)
4. Create representations:
   - TF-IDF then LSA/SVD (optionally scaling)
   - Embeddings (SentenceTransformer)
5. Cluster document vectors (e.g., K-Means)
6. Evaluate each configuration using internal indices
7. Aggregate results, generate plots, export CSV

## Requirements

Typical dependencies used by the notebook:

- Python 3.9+ (recommended)
- pandas, numpy
- scikit-learn
- nltk
- sentence-transformers (for embeddings)
- matplotlib (for plots)

If running in Google Colab, the notebook installs required packages using `pip`.

## Setup

### Option A: Google Colab (recommended)
1. Upload `Financial_document_clustering.ipynb` to Colab.
2. Mount Google Drive (the notebook includes mounting code).
3. Update the CSV path to your dataset.
4. Run all cells.

### Option B: Local environment
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -U pandas numpy scikit-learn nltk matplotlib sentence-transformers
