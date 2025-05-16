import os
import json
import requests
import time
import signal
import logging
from tqdm import tqdm
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

FASTA_PATH = "/home/onur/Desktop/Project/data/mobidb_silver_clustered_40"
OUTPUT_JSON = "/home/onur/Desktop/Project/proj/databases/uniprot/signal_peptide/signal_peptide_labels.json"
LOG_FILE = "/home/onur/Desktop/Project/proj/databases/uniprot/signal_peptide/signal_peptide_failures.txt"
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{}.json"

WRITE_CHUNK_SIZE = 1000
NUM_WORKERS = 4

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

def read_fasta_headers(fasta_path):
    accs = []
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                acc = line[1:].split()[0]
                accs.append(acc)
    return accs

def get_processed_accs(output_path):
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return set(), []
    try:
        with open(output_path, "r") as f:
            data = json.load(f)
            processed = set(entry for d in data for entry in d)
            return processed, data
    except json.JSONDecodeError:
        print(f"[Warning] Failed to load JSON from {output_path}. File might be empty or corrupted.")
        return set(), []

def extract_signal_peptide(acc):
    for attempt in range(5):
        try:
            r = requests.get(UNIPROT_URL.format(acc), timeout=15)
            if r.status_code == 200:
                entry = r.json()
                seq = entry.get("sequence", {}).get("value", "")
                if not seq:
                    return None

                labels = [0] * len(seq)
                for feature in entry.get("features", []):
                    if feature.get("type") == "Signal":
                        loc = feature.get("location", {})
                        start = loc.get("start", {}).get("value")
                        end = loc.get("end", {}).get("value")
                        if start and end and start <= end <= len(seq):
                            for i in range(start - 1, end):
                                labels[i] = 1
                if len(labels) == len(seq):
                    return {acc: labels}
                else:
                    logging.info(f"Length mismatch for {acc}: sequence {len(seq)}, labels {len(labels)}")
                    return None
            else:
                logging.info(f"Failed request for {acc}: Status {r.status_code}")
        except Exception as e:
            logging.info(f"Attempt {attempt+1} failed for {acc}: {e}")
            time.sleep(2)
    logging.info(f"Failed to fetch after retries: {acc}")
    return None

def writer_thread(queue, output_path, initial_data):
    buffer = []
    data = initial_data
    total_written = len(data)
    new_since_last_flush = 0
    flush_count = 0

    while True:
        item = queue.get()
        if item is None:
            break
        data.append(item)
        new_since_last_flush += 1
        total_written += 1

        if new_since_last_flush >= WRITE_CHUNK_SIZE:
            with open(output_path + ".tmp", "w") as f:
                json.dump(data, f, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            os.replace(output_path + ".tmp", output_path)
            flush_count += 1
            print(f"[Writer] Flushed {new_since_last_flush} sequences (flush #{flush_count}). Total written: {total_written}")
            new_since_last_flush = 0
            buffer.clear()

    if buffer:
        data.extend(buffer)
        with open(output_path + ".tmp", "w") as f:
            json.dump(data, f, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(output_path + ".tmp", output_path)
        print(f"[Writer] Final flush to {output_path}: {len(buffer)} remaining. Total written: {total_written}")

def main():
    accs = read_fasta_headers(FASTA_PATH)
    processed, data = get_processed_accs(OUTPUT_JSON)
    unprocessed = [acc for acc in accs if acc not in processed]

    print(f"Total: {len(accs)}, Processed: {len(processed)}, Remaining: {len(unprocessed)}")

    queue = Queue()
    writer = Thread(target=writer_thread, args=(queue, OUTPUT_JSON, data))
    writer.start()

    shutdown_requested = False

    def handle_interrupt(signum, frame):
        nonlocal shutdown_requested
        print("\n[Interrupt] Caught Ctrl+C. Preparing to shut down...")
        shutdown_requested = True

    signal.signal(signal.SIGINT, handle_interrupt)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(extract_signal_peptide, acc): acc for acc in unprocessed}
        try:
            for future in tqdm(futures, desc="Fetching signal peptides"):
                if shutdown_requested:
                    print("[Shutdown] Cancellation requested. Breaking loop.")
                    break
                result = future.result()
                if result:
                    queue.put(result)
        except KeyboardInterrupt:
            print("[KeyboardInterrupt] Halting execution.")
        finally:
            for future in futures:
                future.cancel()

    queue.put(None)
    writer.join()
    print("[Done] Script finished cleanly.")

if __name__ == "__main__":
    main()