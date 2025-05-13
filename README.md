# Interpreting Disordered Proteins using Sparse Auto-Encoders and pLMs

This repository contains the code and documentation for a project aimed at interpreting disordered proteins by leveraging Sparse Autoencoders (SAEs) on embeddings from the ProtT5 protein language model (pLM). The goal is to extract and visualize features of disordered proteins that are not easily derived from sequence data alone, inspired by the reference paper: From Mechanistic Interpretability to Mechanistic Biology bioRxiv. The codebase adapts and extends the architecture from etowahadams/interprot.

### Project Aim
The project focuses on:
- Developing a Sparse Autoencoder (SAE) tailored for the ProtT5 language model to capture latent features of disordered proteins.

- Constructing datasets for protein properties (e.g., disorder regions, phase separation, subcellular localization) from sources like MobiDB, UniProt, and PDB.

- Probing SAE neurons to identify sequence positions associated with specific protein properties.

- Visualizing these features to understand how ProtT5 encodes disordered protein characteristics.

- Integrating the pipeline into the FELLS web server for broader accessibility.


### Dataset
Source: A dataset of 280,589 protein sequences in FASTA format.

Preprocessing:
- Verified uniqueness of protein IDs.

Conducted exploratory data analysis (EDA) to compute:
- Minimum, maximum, average, and median sequence lengths.

Distribution of amino acid sequence lengths.

Modified sequences for ProtT5 compatibility:
- Inserted whitespaces between amino acids.

Replaced rare amino acids (U, Z, O, B) with 'X'.

### Methodology
1. ProtT5 Embeddings Extraction
Model: Used the ProtT5-XL-Half-UniRef50-Enc model for generating per-residue embeddings.

Grid Search for Optimization:
Sampled 1,000 random proteins to optimize embedding extraction.

Parameters tested:
Queue sizes: [50, 100, 500]

Batch sizes: [25, 50, 100, 250]

Compression methods: [gzip, lzf]

Threading: [True, False]

Total configurations: 48 (3 × 4 × 2 × 2).

Selected the best configuration based on performance metrics (e.g., runtime, memory efficiency).

Embedding Extraction:
Processed one sequence at a time to extract embeddings from:
Layer 16 (middle layer): Saved as layer_16.h5 (235 GB).

Layer 24 (output layer): Saved as layer_24.h5 (470 GB).

Total storage: 707 GB.

Maintained a log file with sequence IDs to track extraction, detect duplications, and identify missing proteins by cross-referencing protein IDs.

2. Sparse Autoencoder (SAE) Implementation
Architecture: Adapted the SAE architecture from etowahadams/interprot to suit ProtT5 embeddings.

Grid Search for SAE Training:
Used 2,000 sequences to optimize SAE hyperparameters.

Parameters tested:
Sparsity (k): [16, 32, 64, 128, 256]

Hidden dimensions (d_hidden): [2048, 4096, 8192, 16384]

Layers: [16, 24]

Learning rates: [1e-4, 2e-4, 5e-4]

Total configurations: 120 (5 × 4 × 2 × 3).

Tracked performance using Weights & Biases (WandB) and saved results in CSV files for analysis.

Training: Implemented SAE to learn sparse representations of ProtT5 embeddings, enabling feature extraction for disordered protein properties.

3. Dataset Construction (In Progress)
Sources:
MobiDB: Extracted protein properties (e.g., disorder regions, phase separation, compactness) from TSV and JSON formats.

UniProt: Retrieved signal peptides and transmembrane regions via JSON API.

PDB/MobiDB: Planned extraction of secondary structure data from MobiDB’s MongoDB database.

Additional properties (e.g., aggregation, phosphorylation) to be sourced from PASTA3 and Scop3P.

Properties of Interest:
Disorder regions (curated and homology-based from MobiDB).

Phase separation (curated and homology-based).

Fold-upon-binding regions (filtered for regions ≥10 residues).

Subcellular localization (multi-label classification from MobiDB JSON).

Signal peptides and transmembrane regions (UniProt).

Secondary structure, aggregation, and phosphorylation (to be implemented).

Redundancy Reduction: All datasets will be filtered to ensure ≤25% sequence identity.

4. Probing and Visualization (Planned)
Probe SAE neurons to identify which fire for specific protein properties.

Map active neurons to sequence positions to visualize regions associated with disordered protein features.

Integrate results into the FELLS web server for interactive visualization.





### Steps have been taken:
- Given the dataset contains 280.589 sequences from different cells from different cells, in fasta format file.  
- Explatory data analysis on the  dataset which to see if all protein id`s are uniques, the min-max-avg-median lengths of the datasets, distribution of the lengths of aminoacids.   
- Modified the dataset by adding whitespaces between aminoacids, replacing rare aminoacids (UZOB) with X, which is required for prot-t5 inputs.
- Downloaded and saved the prot_t5_xl_half_uniref50-enc model
- Applied a grid search with 1000 randomly selected proteins for finding best training setup. The grid search was based in queue sizes [50, 100,500], batch sizes[25, 50,100,250] , compression methods [gzip, lzf] and thread [true,false]. In total (3*4*2*2) 48 setup is grid searched. 
- Per-residue embeddings are extracted by feeding prot-t5 one sequence at a tine using the best parameters found via gridsearch from layer 16 (middle layer) and layer 24(output layer). They saved into different hdf5 files based on layers (layer_16.h5 (235 gb) and layer_24.h5 (470 gb)). In total 707 gb.
- While extracting the embeddings log file is used with sequence id`s in case of any corruption or unpredictable things may happen. Using the log file duplications and missing proteins are checked by comparing the file via protein id``s.
- Sae is implemented by modfying and adapting interprot`s architecture to our architecture.
- Grid search is done on SAE using 2000 sequences to find best setup with (sparsity: k (16,32,64,128,256), hidden dimensions: d_hiddens (2048,4096,8192,16384) , layers (16,24) and learning rates (1e-4, 2e-4, 5e-4). In total (5*4*2*3) 120 different models are tested, wanddb is used for tracking the performance of those models as well as csv files are used while training.  
