import os
import re
import time
import h5py
import torch
from tqdm import tqdm
from contextlib import ExitStack
from itertools import islice
from transformers import T5Tokenizer, T5EncoderModel
from queue import Queue
import threading
import numpy as np
import shutil
import csv
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from datetime import datetime


FILE_PATH = "/home/onubac/Project/cleaned_mobidb_silver_clustered_40"
MODEL_PATH = "/home/onubac/Project/model"
OUTPUT_DIR = "/home/onubac/Project/layer_embeddings/"
PER_PROTEIN_FILE = os.path.join(OUTPUT_DIR, "per_protein_embeddings.h5")
LOG_FILE = os.path.join(OUTPUT_DIR, "processed_sequences.log")
SELECTED_LAYERS = list(range(12, 25))  # inclusive: layers 12-24


MAX_SEQUENCES = None
QUEUE_SIZE = 500
BATCH_WRITE_SIZE = 250
COMPRESSION = "lzf"
USE_THREADING = False


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def load_model_and_tokenizer(model_path, device):
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=True)
        model = T5EncoderModel.from_pretrained(model_path, torch_dtype=torch.float16).to(device) # Reduce GPU usage, speeds inference 
        model.eval()
        print(" Model and tokenizer loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f" Failed to load model or tokenizer: {e}")
        exit(1)


def stream_fasta(filepath: str):
    with open(filepath, "r") as file:
        protein_id = None
        sequence_lines = []
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if protein_id is not None:
                    yield protein_id, ''.join(sequence_lines)
                protein_id = line[1:]
                sequence_lines = []
            else:
                sequence_lines.append(line)
        if protein_id is not None:
            yield protein_id, ''.join(sequence_lines)    # returns (protein id, sequence) as string 



def log_processed_protein(protein_id):
    with open(LOG_FILE, "a") as log:
        log.write(protein_id + "\n")



def load_processed_proteins():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as log:
            return set(line.strip() for line in log)
    return set()


# Reads protein sequences, clean and tokenize them and push tokenized sequences into queue for main thread to process with model 
def producer(queue, fasta_path, tokenizer, processed_ids, max_sequences):
    count = 0
    for protein_id, sequence in stream_fasta(fasta_path):       
        if max_sequences and count >= max_sequences:
            break
        if protein_id in processed_ids:
            continue

        tokenized = tokenizer(
            [" ".join(sequence)],  # space-separated amino acids
            add_special_tokens=True,  # from hugging face 
            padding="longest",
            return_tensors="pt"          # pytorch tensors 
        )
        queue.put((protein_id, tokenized))
        count += 1
    queue.put(None)  # signal end





# For isolating writes for per HDF5 file 
def write_one_layer(file, data_batch, compression):
    for protein_id, embedding in data_batch:
        file.create_dataset(protein_id, data=embedding, compression=compression)



# layer_files : {layer: hp5y.file} for each layer 
# per_protein_file : single .h5 file for per-protein embeddings
# layer_batch: dict[layer] = [(id1, emb1), (id2, emb2) ]
# protein_batch: list of (id, per_protein_embedding)
# For grid search we have parallel and serial versions to find faster 
def write_batch_to_disk(layer_files, per_protein_file, layer_batch, protein_batch, use_threading=True):
    if use_threading:
        with ThreadPoolExecutor() as executor:
            futures = []
            for layer, data in layer_batch.items():
                futures.append(executor.submit(write_one_layer, layer_files[layer], data, COMPRESSION))  # for layers 
            futures.append(executor.submit(write_one_layer, per_protein_file, protein_batch, COMPRESSION)) # per protein embedding 
            for f in futures: # Waits for all threads to finish 
                f.result()
    else:
        for layer, data in layer_batch.items():
            write_one_layer(layer_files[layer], data, COMPRESSION)
        write_one_layer(per_protein_file, protein_batch, COMPRESSION)




def extract_and_save_embeddings(fasta_path, model, tokenizer, output_dir, selected_layers, device, max_sequences=None):
    ensure_output_dir(output_dir)
    processed_ids = load_processed_proteins()
    sequence_queue = Queue(maxsize=QUEUE_SIZE)

    with ExitStack() as stack: # Opens hdf5 files, exitstack() makes sure all files closed properly 
        layer_files = {
            layer: stack.enter_context(
                h5py.File(os.path.join(output_dir, f"layer{layer}_embeddings.h5"), "a")
            ) for layer in selected_layers
        }
        per_protein_file = stack.enter_context(h5py.File(PER_PROTEIN_FILE, "a"))

        producer_thread = threading.Thread(          # launchs thread to read and tokenize sequences beforehand  for main thread 
            target=producer,
            args=(sequence_queue, fasta_path, tokenizer, processed_ids, max_sequences),
            daemon=True
        )
        producer_thread.start()

        index = 0
        layer_batch = {layer: [] for layer in selected_layers}      # stores per layer embeddings 
        protein_batch = []                                          # stores per-protein embedding                       

        tqdm_total = max_sequences or 0
        pbar = tqdm(total=tqdm_total, desc="Processing Sequences")

        while True:
            item = sequence_queue.get()
            if item is None:
                break

            protein_id, tokenized = item
            input_ids = tokenized['input_ids'].to(device)          
            attention_mask = tokenized['attention_mask'].to(device) 

            try:
                with torch.no_grad():                           
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                    hidden_states = outputs.hidden_states

                for layer in selected_layers:                                                 # Collect embeddings 
                    embedding = hidden_states[layer][0].cpu().numpy()
                    layer_batch[layer].append((protein_id, embedding))

                per_protein_embedding = hidden_states[24][0].mean(dim=0).cpu().numpy()
                protein_batch.append((protein_id, per_protein_embedding))

                log_processed_protein(protein_id)                                         # For logs 
                index += 1
                pbar.update(1)

                if index % BATCH_WRITE_SIZE == 0:
                    write_batch_to_disk(layer_files, per_protein_file, layer_batch, protein_batch, use_threading=USE_THREADING)
                    layer_batch = {layer: [] for layer in selected_layers}
                    protein_batch = []

            except Exception as e:
                print(f" Error processing {protein_id}: {e}")

        if any(layer_batch.values()) or protein_batch:                                  # Remaining data did not fill a full batch 
            write_batch_to_disk(layer_files, per_protein_file, layer_batch, protein_batch, use_threading=USE_THREADING)
        pbar.close()






def main():
    print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_output_dir(OUTPUT_DIR)
    tokenizer, model = load_model_and_tokenizer(MODEL_PATH, device)

    if tokenizer is None or model is None:
        print(" Exiting due to failed model/tokenizer load.")
        return
    start = time.time()
    extract_and_save_embeddings(
        fasta_path=FILE_PATH,
        model=model,
        tokenizer=tokenizer,
        output_dir=OUTPUT_DIR,
        selected_layers=SELECTED_LAYERS,
        device=device,
        max_sequences=MAX_SEQUENCES
    )
    end = time.time()
    print(f"\n Done. Time taken: {end - start:.2f} seconds")



if __name__ == "__main__":
    main()