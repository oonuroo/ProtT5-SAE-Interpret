# Interpreting Disordered Proteins Using Sparse Autoencoders and Protein Language Models 🧬

This repository contains code and documentation for interpreting **disordered proteins** using **Sparse Autoencoders (SAEs)** on embeddings from the **ProtT5** protein language model (pLM). Inspired by *From Mechanistic Interpretability to Mechanistic Biology* ([bioRxiv](https://www.biorxiv.org/content/10.1101/2024.08.27.609883v1)), it adapts the architecture from [etowahadams/interprot](https://github.com/etowahadams/interprot) to extract and visualize latent features of disordered proteins.

## Project Aim 🎯
The goal is to:
- Develop an SAE for ProtT5 to capture features of disordered proteins beyond sequence-derived properties.
- Construct datasets for protein properties (e.g., disorder regions, phase separation) from MobiDB, UniProt, and PDB.
- Probe SAE neurons to identify sequence positions linked to these properties.
- Visualize feature-associated regions to understand ProtT5's encoding.
- Integrate the pipeline into the [FELLS web server](https://fells.org/).

## Dataset 📊
- **Source**: **280,589 protein sequences** in FASTA format, sourced from [UniProt](https://www.uniprot.org/) with diverse cellular contexts.
- **Preprocessing**:
  - Verified unique protein IDs.
  - Performed exploratory data analysis (EDA):
    - Sequence length stats: min, max, average, median.
    - Amino acid length distribution.
  - Prepared for ProtT5:
    - Added whitespaces between amino acids.
    - Replaced rare amino acids (U, Z, O, B) with 'X'.
- **Redundancy Reduction**: In progress, targeting ≤25% sequence identity using CD-HIT.
- **Note**: Large dataset size (707 GB embeddings) required single-sequence processing to manage memory.

## Methodology ⚙️

### 1. ProtT5 Embeddings Extraction
- **Model**: `ProtT5-XL-Half-UniRef50-Enc` ([Hugging Face](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc)).
- **Grid Search**:
  - Tested 1,000 random proteins to optimize extraction.
  - Parameters:
    - Queue sizes: `[50, 100, 500]`
    - Batch sizes: `[25, 50, 100, 250]`
    - Compression: `[gzip, lzf]`
    - Threading: `[True, False]`
  - Total: **48** configurations (3 × 4 × 2 × 2).
  - Best setup chosen based on runtime and memory efficiency.
- **Extraction**:
  - Per-residue embeddings from:
    - **Layer 16** (middle): `layer_16.h5` (235 GB).
    - **Layer 24** (output): `layer_24.h5` (470 GB).
  - Total: **707 GB**.
  - Logged sequence IDs to ensure data integrity (no duplications or missing proteins).

### 2. Sparse Autoencoder (SAE) Implementation
- **Architecture**: Adapted from [etowahadams/interprot](https://github.com/etowahadams/interprot) for ProtT5 embeddings.
- **Grid Search**:
  - Trained on 2,000 sequences.
  - Parameters:
    - Sparsity (k): `[16, 32, 64, 128, 256]`
    - Hidden dimensions: `[2048, 4096, 8192, 16384]`
    - Layers: `[16, 24]`
    - Learning rates: `[1e-4, 2e-4, 5e-4]`
  - Total: **120** configurations (5 × 4 × 2 × 3).
  - Evaluated via reconstruction loss and sparsity, tracked with [Weights & Biases (WandB)](https://wandb.ai/) and CSV logs.
- **Training**: SAE learns sparse representations for feature extraction.

### 3. Dataset Construction (In Progress) 📋
- **Sources**:
  - **MobiDB**: Disorder regions, phase separation, compactness ([TSV](https://mobidb.org/api/download?acc=P04637&format=tsv), [JSON](https://mobidb.org/)).
  - **UniProt**: Signal peptides, transmembrane regions ([API](https://rest.uniprot.org/)).
  - **PDB/MobiDB**: Secondary structure (planned, via MobiDB MongoDB).
  - **Other**: Aggregation (PASTA3), phosphorylation (Scop3P).
- **Properties**:
  - Disorder regions, phase separation, fold-upon-binding (≥10 residues), subcellular localization, signal peptides, transmembrane regions, secondary structure, aggregation, phosphorylation.
- **Redundancy**: Targeting ≤25% sequence identity.

### 4. Probing and Visualization (Planned) 📈
- Probe SAE neurons for property-specific signals.
- Map active neurons to sequence positions.
- Integrate into FELLS for interactive visualization.

## Repository Structure
