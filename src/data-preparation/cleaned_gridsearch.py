import os
import re
import time
import h5py
import torch
import csv
import shutil
from tqdm import tqdm
from queue import Queue
from datetime import datetime
from itertools import product
from threading import Thread
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from transformers import T5Tokenizer, T5EncoderModel

# Paths
FILE_PATH = "/home/onur/Desktop/Project/cleaned_mobidb_silver_clustered_40"
MODEL_PATH = "/home/onur/models/prot_t5_local"
OUTPUT_DIR = "/home/onur/Desktop/Project/layer_embeddings_cleaned/"
PER_PROTEIN_FILE = os.path.join(OUTPUT_DIR, "per_protein_embeddings.h5")
LOG_FILE = os.path.join(OUTPUT_DIR, "processed_sequences.log")

# Static config
SELECTED_LAYERS = list(range(12, 25))
MAX_SEQUENCES = 500

def clean_sequence(seq: str) -> str:
    return " ".join(list(re.sub(r"[UZOB]", "X", seq)))

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

def log_processed_protein(protein_id):
    with open(LOG_FILE, "a") as log:
        log.write(protein_id + "\n")

def load_processed_proteins():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as log:
            return set(line.strip() for line in log)
    return set()

def reset_output_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def load_model_and_tokenizer(model_path, device):
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=True)
    model = T5EncoderModel.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    model.eval()
    return tokenizer, model

def producer(queue, fasta_path, tokenizer, processed_ids, max_sequences):
    count = 0
    for protein_id, sequence in stream_fasta(fasta_path):
        if max_sequences and count >= max_sequences:
            break
        if protein_id in processed_ids:
            continue
        tokenized = tokenizer.batch_encode_plus(
            [sequence],
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt"
        )
        queue.put((protein_id, tokenized))
        count += 1
    queue.put(None)

def write_one_layer(file, data_batch, compression):
    for protein_id, embedding in data_batch:
        file.create_dataset(protein_id, data=embedding, compression=compression)

def write_batch_to_disk(layer_files, per_protein_file, layer_batch, protein_batch, compression, threaded):
    if threaded:
        with ThreadPoolExecutor() as executor:
            futures = []
            for layer, data in layer_batch.items():
                futures.append(executor.submit(write_one_layer, layer_files[layer], data, compression))
            futures.append(executor.submit(write_one_layer, per_protein_file, protein_batch, compression))
            for f in futures:
                f.result()
    else:
        for layer, data in layer_batch.items():
            write_one_layer(layer_files[layer], data, compression)
        write_one_layer(per_protein_file, protein_batch, compression)

def run_embedding_pipeline(queue_size, batch_size, compression, threaded):
    reset_output_dir(OUTPUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model_and_tokenizer(MODEL_PATH, device)

    processed_ids = load_processed_proteins()
    sequence_queue = Queue(maxsize=queue_size)

    with ExitStack() as stack:
        layer_files = {
            layer: stack.enter_context(
                h5py.File(os.path.join(OUTPUT_DIR, f"layer{layer}_embeddings.h5"), "a")
            ) for layer in SELECTED_LAYERS
        }
        per_protein_file = stack.enter_context(h5py.File(PER_PROTEIN_FILE, "a"))

        producer_thread = Thread(
            target=producer,
            args=(sequence_queue, FILE_PATH, tokenizer, processed_ids, MAX_SEQUENCES),
            daemon=True
        )
        producer_thread.start()

        index = 0
        layer_batch = {layer: [] for layer in SELECTED_LAYERS}
        protein_batch = []
        pbar = tqdm(total=MAX_SEQUENCES, desc="Processing")

        while True:
            item = sequence_queue.get()
            if item is None:
                break

            protein_id, tokenized = item
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)

            try:
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    hidden_states = outputs.hidden_states

                for layer in SELECTED_LAYERS:
                    embedding = hidden_states[layer][0].cpu().numpy()
                    layer_batch[layer].append((protein_id, embedding))

                per_protein_embedding = hidden_states[24][0].mean(dim=0).cpu().numpy()
                protein_batch.append((protein_id, per_protein_embedding))

                log_processed_protein(protein_id)
                index += 1
                pbar.update(1)

                if index % batch_size == 0:
                    write_batch_to_disk(layer_files, per_protein_file, layer_batch, protein_batch, compression, threaded)
                    layer_batch = {layer: [] for layer in SELECTED_LAYERS}
                    protein_batch = []

            except Exception as e:
                print(f" Error with {protein_id}: {e}")

        if any(layer_batch.values()) or protein_batch:
            write_batch_to_disk(layer_files, per_protein_file, layer_batch, protein_batch, compression, threaded)
        pbar.close()

# Grid Search Setup
def grid_search():
    queue_sizes = [ 50, 100,500]
    batch_sizes = [ 25, 50,100,250]
    compressions = ['gzip', 'lzf']
    threading_options = [True, False]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"grid_search_cleaned_{timestamp}.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Run", "QueueSize", "BatchSize", "Compression", "ThreadedWrite", "DurationSeconds"])

        total_runs = len(queue_sizes) * len(batch_sizes) * len(compressions) * len(threading_options)
        run_id = 1

        for queue_size, batch_size, compression, threaded in product(queue_sizes, batch_sizes, compressions, threading_options):
            print(f"\n🧪 Grid Run {run_id}/{total_runs}")
            print(f"🔧 QUEUE={queue_size}, BATCH={batch_size}, COMP={compression}, THREADS={threaded}")

            start = time.time()
            run_embedding_pipeline(queue_size, batch_size, compression, threaded)
            duration = time.time() - start

            writer.writerow([run_id, queue_size, batch_size, compression, threaded, round(duration, 2)])
            run_id += 1

    print(f"\nResults saved to {csv_file}")

# Run the grid search
if __name__ == "__main__":
    grid_search()
