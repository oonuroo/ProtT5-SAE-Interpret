Interpreting Disordered Proteins Using Sparse Autoencoders and Protein Language Models 
This repository hosts the code and documentation for a project focused on interpreting disordered proteins using Sparse Autoencoders (SAEs) applied to embeddings from the ProtT5 protein language model (pLM). Inspired by From Mechanistic Interpretability to Mechanistic Biology bioRxiv, this work adapts the architecture from etowahadams/interprot to extract and visualize latent features of disordered proteins.
Project Aim
The goal is to:
Develop an SAE for ProtT5 to capture features of disordered proteins not easily derived from sequences.

Construct datasets for protein properties (e.g., disorder regions, phase separation) from MobiDB, UniProt, and PDB.

Probe SAE neurons to identify sequence positions linked to these properties.

Visualize feature-associated regions to understand ProtT5's encoding of disordered proteins.

Integrate the pipeline into the FELLS web server for public access.

Dataset 
Source: A dataset of 280,589 protein sequences in FASTA format, sourced from UniProt and filtered for diverse cellular contexts.

Preprocessing:
Confirmed unique protein IDs.

Performed exploratory data analysis (EDA):
Sequence length statistics: min, max, average, median.

Amino acid length distribution.

Prepared sequences for ProtT5:
Added whitespaces between amino acids.

Replaced rare amino acids (U, Z, O, B) with 'X'.

Redundancy Reduction: Sequences will be filtered to ≤25% sequence identity using a clustering algorithm (e.g., CD-HIT, in progress).

Note: The dataset size posed memory challenges during embedding extraction, mitigated by processing sequences individually.

Methodology
1. ProtT5 Embeddings Extraction
Model: Utilized ProtT5-XL-Half-UniRef50-Enc Hugging Face.

Grid Search:
Tested 1,000 randomly sampled proteins to optimize embedding extraction.

Parameters:
Queue sizes: [50, 100, 500]

Batch sizes: [25, 50, 100, 250]

Compression: [gzip, lzf]

Threading: [True, False]

Total configurations: 48 (3 × 4 × 2 × 2).

Selected best setup based on runtime and memory usage.

Embedding Extraction:
Extracted per-residue embeddings from:
Layer 16 (middle): layer_16.h5 (235 GB).

Layer 24 (output): layer_24.h5 (470 GB).

Total storage: 707 GB.

Logged sequence IDs to detect duplications or missing proteins, ensuring data integrity.

2. Sparse Autoencoder (SAE) Implementation
Architecture: Modified etowahadams/interprot to process ProtT5 embeddings.

Grid Search:
Trained on 2,000 sequences to optimize SAE hyperparameters.

Parameters:
Sparsity (k): [16, 32, 64, 128, 256]

Hidden dimensions: [2048, 4096, 8192, 16384]

Layers: [16, 24]

Learning rates: [1e-4, 2e-4, 5e-4]

Total configurations: 120 (5 × 4 × 2 × 3).

Evaluated using reconstruction loss and sparsity metrics, tracked via Weights & Biases (WandB) and CSV logs.

Training: SAE learns sparse representations to enable feature extraction for disordered protein properties.

3. Dataset Construction (In Progress)
Sources:
MobiDB: Disorder regions, phase separation, compactness (TSV, JSON).

UniProt: Signal peptides, transmembrane regions API.

PDB/MobiDB: Secondary structure (to be extracted from MobiDB MongoDB).

Other: Aggregation (PASTA3), phosphorylation (Scop3P).

Properties:
Disorder regions, phase separation, fold-upon-binding (≥10 residues), subcellular localization, signal peptides, transmembrane regions, secondary structure, aggregation, phosphorylation.

Redundancy: Datasets will be reduced to ≤25% sequence identity.

4. Probing and Visualization (Planned)
Probe SAE neurons to identify signals for specific protein properties.

Map active neurons to sequence positions for visualization.

Integrate into FELLS for interactive feature exploration.

Repository Structure

├── data/                   # Raw and preprocessed datasets
├── embeddings/             # ProtT5 embeddings (layer_16.h5, layer_24.h5)
├── models/                 # Trained SAE models
├── scripts/                # Preprocessing, embedding extraction, SAE training
├── logs/                   # Extraction and training logs
├── results/                # Grid search results (CSV, WandB)
├── docs/                   # Pipeline diagram, additional docs
└── README.md               # Project overview

Installation
Clone the repository:
bash

git clone https://github.com/<username>/ProtT5-SAE-Interpret.git
cd ProtT5-SAE-Interpret

Install dependencies:
bash

pip install -r requirements.txt

Download ProtT5 weights (see scripts/setup_model.py).

Usage
Preprocessing: scripts/preprocess_fasta.py

Embedding Extraction: scripts/extract_embeddings.py

SAE Training: scripts/train_sae.py

Results: View grid search outputs in results/ and WandB.

Next Steps
Finalize dataset construction for all properties.

Implement neuron probing and visualization.

Integrate with FELLS web server.

Explore ProtT5 fine-tuning if feature capture is suboptimal.

Dependencies
Python 3.8+

PyTorch, Transformers, h5py, WandB, NumPy, Pandas, Biopython

See requirements.txt for details.

License
This project is licensed under the MIT License. See LICENSE for details.
Contributing
Contributions are welcome! Please:
Fork the repository.

Create a feature branch (git checkout -b feature-name).

Submit a pull request with a clear description of changes.
For questions, open an issue or contact [<your-email>].

References
Adams, E., et al. (2024). From Mechanistic Interpretability to Mechanistic Biology. bioRxiv.

ProtT5: Hugging Face.

MobiDB: mobidb.org.

UniProt: rest.uniprot.org.

Contact
For inquiries, contact [<your-email>] or open a GitHub issue.

