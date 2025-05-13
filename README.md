# Interpreting Disordered Proteins using Sparse Auto-Encoders and pLMs

This repository contains the code and documentation for a project aimed at interpreting disordered proteins by leveraging Sparse Autoencoders (SAEs) on embeddings from the ProtT5 protein language model (pLM). The goal is to extract and visualize features of disordered proteins that are not easily derived from sequence data alone, inspired by the reference paper: From Mechanistic Interpretability to Mechanistic Biology bioRxiv. The codebase adapts and extends the architecture from etowahadams/interprot.

### Project Aim
The project focuses on:
- Developing a Sparse Autoencoder (SAE) tailored for the ProtT5 language model to capture latent features of disordered proteins.

- Constructing datasets for protein properties (e.g., disorder regions, phase separation, subcellular localization) from sources like MobiDB, UniProt, and PDB.

- Probing SAE neurons to identify sequence positions associated with specific protein properties.

- Visualizing these features to understand how ProtT5 encodes disordered protein characteristics.

- Integrating the pipeline into the FELLS web server for broader accessibility.





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
