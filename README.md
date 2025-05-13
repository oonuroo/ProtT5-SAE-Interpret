# Interpreting Disordered Proteins using Sparse Auto-Encoders and pLMs

This repository contains the code and documentation for a project aimed at interpreting disordered proteins by leveraging Sparse Autoencoders (SAEs) on embeddings from the ProtT5 protein language model (pLM). The goal is to extract and visualize features of disordered proteins that are not easily derived from sequence data alone, inspired by the reference paper: From Mechanistic Interpretability to Mechanistic Biology bioRxiv. The codebase adapts and extends the architecture from etowahadams/interprot.

## Project Aim
The project focuses on:
- Developing a Sparse Autoencoder (SAE) tailored for the ProtT5 language model to capture latent features of disordered proteins.

- Constructing datasets for protein properties (e.g., disorder regions, phase separation, subcellular localization) from sources like MobiDB, UniProt, and PDB.

- Probing SAE neurons to identify sequence positions associated with specific protein properties.

- Visualizing these features to understand how ProtT5 encodes disordered protein characteristics.

- Integrating the pipeline into the FELLS web server for broader accessibility.


## Dataset
Source: A dataset of 280,589 protein sequences in FASTA format.

Preprocessing:
- Verified uniqueness of protein IDs.

Conducted exploratory data analysis (EDA) to compute:
- Minimum, maximum, average, and median sequence lengths.

Distribution of amino acid sequence lengths.

Modified sequences for ProtT5 compatibility:
- Inserted whitespaces between amino acids.

Replaced rare amino acids (U, Z, O, B) with 'X'.

## Methodology
### 1. ProtT5 Embeddings Extraction
Model: Used the ProtT5-XL-Half-UniRef50-Enc model for generating per-residue embeddings.

#### Grid Search for Optimization:
Sampled 1,000 random proteins to optimize embedding extraction.

#### Parameters tested: 
Total configurations: 48 (3 × 4 × 2 × 2).
- Queue sizes: [50, 100, 500]

- Batch sizes: [25, 50, 100, 250]

- Compression methods: [gzip, lzf]

- Threading: [True, False]

Selected the best configuration based on performance metrics (e.g., runtime, memory efficiency).

#### Embedding Extraction:
Processed one sequence at a time to extract embeddings from:
- Layer 16 (middle layer): Saved as layer_16.h5 (235 GB).

- Layer 24 (output layer): Saved as layer_24.h5 (470 GB).
Total storage: 707 GB.

Maintained a log file with sequence IDs to track extraction, detect duplications, and identify missing proteins by cross-referencing protein IDs.

### 2. Sparse Autoencoder (SAE) Implementation
Architecture: Adapted the SAE architecture from etowahadams/interprot to suit ProtT5 embeddings.

#### Grid Search for SAE Training:
Used 2,000 sequences to optimize SAE hyperparameters.

#### Parameters tested:
Total configurations: 120 (5 × 4 × 2 × 3).
- Sparsity (k): [16, 32, 64, 128, 256]

- Hidden dimensions (d_hidden): [2048, 4096, 8192, 16384]

- Layers: [16, 24]

- Learning rates: [1e-4, 2e-4, 5e-4]

Tracked performance using Weights & Biases (WandB) and also saved results in CSV files for analysis.

#### Training: Implemented SAE to learn sparse representations of ProtT5 embeddings, enabling feature extraction for disordered protein properties.

### 3. Dataset Construction (In Progress)
##### Sources:
- MobiDB: Extracted protein properties (e.g., disorder regions, phase separation, compactness) from TSV and JSON formats.

- UniProt: Retrieved signal peptides and transmembrane regions via JSON API.

- PDB/MobiDB: Planned extraction of secondary structure data from MobiDB’s MongoDB database.

- Additional properties (e.g., aggregation, phosphorylation) to be sourced from PASTA3 and Scop3P.

##### Properties of Interest:
- 1. Disorder regions (curated and homology-based from MobiDB).

- 2. Phase separation (curated and homology-based).

- 3. Fold-upon-binding regions (filtered for regions ≥10 residues).

- 4. Subcellular localization (multi-label classification from MobiDB JSON).

- 5. Signal peptides and transmembrane regions (UniProt).

- 6. Secondary structure, aggregation, and phosphorylation (to be implemented).



### 4. Probing and Visualization (Planned)
- Probe SAE neurons to identify which fire for specific protein properties.

- Map active neurons to sequence positions to visualize regions associated with disordered protein features.

- Integrate results into the FELLS web server for interactive visualization.
