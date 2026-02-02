ECG_Leads_Reconstruction

This repository contains a complete pipeline for ECG lead reconstruction using contrastive representation learning and lightweight decoding models.
The workflow is designed around the PTB-XL dataset and enforces strict patient-wise separation across all stages.

Overview

The pipeline consists of four main stages:

ECG signal cleaning and standardization

Sliding-window segmentation with quality control

Contrastive representation learning and extraction

Patient-wise training and evaluation of lead reconstruction models

All large datasets, generated segments, and model artifacts are intentionally excluded from this repository.

Data Cleaning and Standardization

The script Cleaning_Two_Datasets.py performs preprocessing of raw PTB-XL ECG recordings.

Signals are filtered using standard signal-processing techniques to remove baseline wander and high-frequency noise, resampled to a fixed temporal resolution, and standardized to a consistent 12-lead format. Each cleaned record is saved as a NumPy array, accompanied by a JSON file containing patient identifiers and diagnostic metadata.

Cleaned outputs are stored under:

Cleaned_Datasets/ptbxl_clean/

Sliding-Window Segmentation

The script Segmentation.py segments cleaned ECG recordings into fixed-length overlapping windows suitable for learning-based models.

A two-pass strategy is used:

In the first pass, per-lead amplitude and RMS statistics are estimated from sampled segments.

In the second pass, sliding-window segments are extracted and filtered using percentile-based quality-control thresholds.

Accepted segments are written in sharded NumPy files with aligned metadata, along with summary statistics and quality-control reports.

Outputs are stored under:

Segments/ptbxl/

Contrastive Representation Learning

The script contrastive_morphology.py trains a 1D convolutional encoder using supervised contrastive learning on segmented ECG data.

Segments are sampled in class-balanced batches, and positive pairs are formed based on shared diagnostic labels. The encoder operates on a subset of ECG leads and learns normalized latent embeddings that capture morphological structure relevant for reconstruction.

Representation Extraction

The script extract_contrastive_reps.py applies the trained contrastive encoder to segmented ECG data and extracts latent representations.

For each segment, a fixed-dimensional latent vector is produced. These representations can optionally be tiled along the temporal axis and are saved in sharded NumPy files together with lightweight identifiers for traceability.

Extracted representations are stored under:

reps/

Patient-wise Data Organization

Two scripts are used to enforce patient-level separation:

make_patient_folds.py generates patient-wise cross-validation folds using group-based splitting, ensuring no patient overlap between training and validation sets.

reorder_shards_by_patient.py reorganizes segmented data into patient-consistent train, validation, and test splits, matching predefined split ratios.

This design prevents information leakage and ensures reliable evaluation.

Lead Reconstruction Models

The script Final_Model_Folds_Simple.py trains and evaluates the final ECG lead reconstruction models.

Multiple input sources, including raw clean signals and contrastive representations, are projected into a shared latent space and fused. A lightweight convolutional decoder is trained separately for each target lead to reconstruct missing ECG leads.

Training is performed using fixed patient-wise folds. Model selection relies on validation loss, followed by evaluation on held-out patient-wise test data. Performance is reported using RMSE, R², and Pearson correlation coefficients.

Outputs, including predictions, trained models, and evaluation summaries, are saved under:

StackedRecons_ptbxl_Contr/

Notes

This repository contains code only.

Raw datasets, segmented data, extracted representations, and trained models are excluded via .gitignore.

The pipeline is designed for reproducible research and can be adapted to other multi-lead ECG datasets with minimal changes.
