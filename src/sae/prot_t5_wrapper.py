import os
import time
import h5py                                                # for hdf5 files 
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel
from queue import Queue
import threading
import numpy as np
import signal

FILE_PATH = "/home/onur/Desktop/Project/data/cleaned_mobidb_silver_clustered_40"
MODEL_PATH = "/home/onur/Desktop/Project/model/prot_t5_xl_half_uniref50-enc"
OUTPUT_DIR = "/home/onur/Desktop/Project/last-embed"

PER_RESIDUE_FILE = os.path.join(OUTPUT_DIR, "output_layer_embeddings.h5")
LOG_FILE = os.path.join("/home/onur/Desktop/Project/last-embed", "processed_sequences.log")
SELECTED_LAYERS = [16]  # Layers wanted to extract [12,13,14,...]


MAX_SEQUENCES = None            # For testing; i.e 100 runs for 100 sequences
QUEUE_SIZE = 500
BATCH_WRITE_SIZE = 250
COMPRESSION = "lzf"
USE_THREADING = False
open_h5_files = []

#Creates output directory if not exists 
def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)

# Opens hsf5 files 
def safe_h5_open(path, mode):
    f = h5py.File(path, mode)
    open_h5_files.append(f)
    return f

# Ensures hdf5 files opened safely and closed in the interruption 
def cleanup_h5_files(*args):
    for f in open_h5_files:
        try:
            f.flush()
            f.close()
        except Exception as e:
            print(f"\u274c Failed to close file: {e}")
    print("\u2705 All HDF5 files closed safely.")
    exit(0)

signal.signal(signal.SIGINT, cleanup_h5_files)
signal.signal(signal.SIGTERM, cleanup_h5_files)

# Loads tokenizer and model from MODEL_PATH defined head of the code 
def load_model_and_tokenizer(model_path, device):
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=True)
        model = T5EncoderModel.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
        model.eval()
        print(" Model and tokenizer loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f" Failed to load model or tokenizer: {e}")
        exit(1)

# Reads the file line by line and yields (protein_id,sequence) pairs 
# Sequences can be multi-line so checks if line starts with > 
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
            yield protein_id, ''.join(sequence_lines)



def count_total_sequences(fasta_path):
    count = 0
    with open(fasta_path, "r") as file:
        for line in file:
            if line.startswith(">"):
                count += 1
    return count

def log_processed_protein(protein_id):
    with open(LOG_FILE, "a") as log:
        log.write(protein_id + "\n")

def load_processed_proteins():
    if os.path.exists(LOG_FILE) and os.path.isfile(LOG_FILE):
        with open(LOG_FILE, "r") as log:
            return set(line.strip() for line in log)
    return set()

# Reads the sequences and skips already processed ones 
# Tokenize sequence into input for make compatible with model
# Adds ready inputs to queue
def producer(queue, fasta_path, tokenizer, processed_ids, max_sequences):
    count = 0
    for protein_id, sequence in stream_fasta(fasta_path):
        if max_sequences and count >= max_sequences:
            break
        if protein_id in processed_ids:
            continue
        tokenized = tokenizer([" ".join(sequence)], add_special_tokens=True, padding="longest", return_tensors="pt")
        queue.put((protein_id, tokenized))
        count += 1
    queue.put(None)

# Writes batches to .h5 file
# Each layer has different file
# Last layer written to output_layer_embeding.h5 
def write_batch_to_disk(output_dir, layer_batch, output_layer_batch, compression):
    for layer, data in layer_batch.items():
        layer_path = os.path.join(output_dir, f"layer{layer}_embeddings.h5")
        with safe_h5_open(layer_path, "a") as f:
            for protein_id, embedding in data:
                if protein_id not in f:
                    f.create_dataset(protein_id, data=embedding, compression=compression)

    with safe_h5_open(PER_RESIDUE_FILE, "a") as f:
        for protein_id, embedding in output_layer_batch:
            if protein_id not in f:
                f.create_dataset(protein_id, data=embedding, compression=compression)

# Creates output directiry
# Loads list of already processed Id`s
# Initialize the queue
# Tdm to keep track of process
# Waits for items from queue
# Move inputs if to attention mask of device
# Extract hidden states
# Log process id`s
# # Every batch writes embeddings to disk

def extract_and_save_embeddings(fasta_path, model, tokenizer, output_dir, selected_layers, device, max_sequences=None):
    ensure_output_dir(output_dir)
    processed_ids = load_processed_proteins()
    sequence_queue = Queue(maxsize=QUEUE_SIZE)

    producer_thread = threading.Thread(
        target=producer,
        args=(sequence_queue, fasta_path, tokenizer, processed_ids, max_sequences),
        daemon=True
    )
    producer_thread.start()

    all_sequences = count_total_sequences(fasta_path)
    tqdm_total = min(max_sequences, all_sequences) if max_sequences else all_sequences
    tqdm_total -= len(processed_ids)

    index = 0
    layer_batch = {layer: [] for layer in selected_layers}
    output_layer_batch = []
    pbar = tqdm(total=tqdm_total, desc="Processing Sequences", unit="seq", dynamic_ncols=True)

    while True:
        item = sequence_queue.get()
        if item is None:
            break

        protein_id, tokenized = item                               # each queue is a tuple (id, sequence)
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)

        try:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states

            for layer in selected_layers:
                embedding = hidden_states[layer][0].cpu().numpy()
                layer_batch[layer].append((protein_id, embedding))

            output_layer_embedding = outputs.last_hidden_state[0].cpu().numpy()
            output_layer_batch.append((protein_id, output_layer_embedding))

            log_processed_protein(protein_id)
            index += 1
            pbar.update(1)

            if index % BATCH_WRITE_SIZE == 0:
                write_batch_to_disk(output_dir, layer_batch, output_layer_batch, COMPRESSION)
                layer_batch = {layer: [] for layer in selected_layers}
                output_layer_batch = []

        except Exception as e:
            print(f" Error processing {protein_id}: {e}")

    if any(layer_batch.values()) or output_layer_batch:
        write_batch_to_disk(output_dir, layer_batch, output_layer_batch, COMPRESSION)
    pbar.close()

# Detect if GPU avaliable
# Loads model and tokenizer
# Run pipeline
# Reports time 
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

