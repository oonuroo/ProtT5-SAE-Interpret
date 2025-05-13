# The aim of the project: Interpreting Disordered Proteins using Sparse Auto-Encoders and pLMs
### Steps have been taken:
- Given the dataset contains 280.589 sequences from different cells from different cells, in fasta format file.  
- Explatory data analysis on the  dataset which to see if all protein id`s are uniques, the min-max-avg-median lengths of the datasets, distribution of the lengths of aminoacids.   
- Modified the dataset by adding whitespaces between aminoacids, replacing rare aminoacids (UZOB) with X, which is required for prot-t5 inputs.
- Downloaded and saved the prot_t5_xl_half_uniref50-enc model
- Applied a grid search with 1000 randomly selected proteins for finding best training setup. The grid search was based in queue sizes [50, 100,500], batch sizes[25, 50,100,250] , compression methods [gzip, lzf] and thread [true,false]. In total (3*4*2*2) 48 setup is grid searched. 
- Per-residue embeddings are extracted by feeding prot-t5 one sequence at a tine using the best parameters found via gridsearch from layer 16 (middle layer) and layer 24(output layer). They saved into different hdf5 files based on layers (layer_16.h5 (235 gb) and layer_24.h5 (470 gb)). In total 707 gb.
- While extracting the embeddings log file is used with sequence id`s in case of any corruption or unpredictable things may happen. Using the log file duplications and missing proteins are checked by comparing the file via protein id``s.
