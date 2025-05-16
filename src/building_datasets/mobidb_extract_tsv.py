# Enhanced script with all requested improvements
import os
import requests
import json
import time
import sys
from tqdm import tqdm
from threading import Thread
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import logging
import signal

# Paths
FASTA_PATH = "/home/onur/Desktop/Project/data/mobidb_silver_clustered_40"
DISORDER_OUTPUT_JSON = "/home/onur/Desktop/Project/proj/databases/tsv/disorder_regions/disorder_labels.json"
PHASESEP_OUTPUT_JSON = "/home/onur/Desktop/Project/proj/databases/tsv/phase_seperation/phase_sep_labels.json"
FOLD_BINDING_OUTPUT_JSON = "/home/onur/Desktop/Project/proj/databases/tsv/fold_upon_binding/fold_binding_labels.json"
ELM_OUTPUT_JSON = "/home/onur/Desktop/Project/proj/databases/tsv/elm/elm_labels.json"
SUBCLASS_POS1_JSON = "/home/onur/Desktop/Project/proj/databases/tsv/sub-classes/pos_1/subclass_pos1.json"
SUBCLASS_POS2_JSON = "/home/onur/Desktop/Project/proj/databases/tsv/sub-classes/pos_2/subclass_pos2.json"

MOBIDB_TSV_URL = "https://mobidb.org/api/download?acc={}&format=tsv"
LOG_FILE = "/home/onur/Desktop/Project/proj/src/building_datasets/failed_proteins.txt"

# Feature sets
FEATURE_MAP = {
    "disorder": {
        "pos": {"curated-disorder-disprot", "homology-disorder-disprot"},
        "neg": {"derived-observed-th_90"},
        "output": DISORDER_OUTPUT_JSON,
    },
    "phase": {
        "pos": {"curated-phase_separation-merge", "homology-phase_separation-merge"},
        "neg": set(),
        "output": PHASESEP_OUTPUT_JSON,
    },
    "fold": {
        "pos": {"curated-lip-dibs", "homology-lip-dibs", "prediction-lip-alphafold"},
        "neg": set(),
        "output": FOLD_BINDING_OUTPUT_JSON,
    },
    "elm": {
        "pos": {"curated-lip-elm", "homology-lip-elm"},
        "neg": set(),
        "output": ELM_OUTPUT_JSON,
    },
    "subclass1": {
        "pos": {"prediction-compact-mobidb_lite_sub"},
        "neg": set(),
        "output": SUBCLASS_POS1_JSON,
    },
    "subclass2": {
        "pos": {"prediction-extended-mobidb_lite_sub"},
        "neg": set(),
        "output": SUBCLASS_POS2_JSON,
    },
}

WRITE_CHUNK_SIZE = 1000
NUM_WORKERS = 2

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(message)s", filemode="a")

def parse_region(region_str, seq_len):
    regions = []
    for part in region_str.split(","):
        part = part.strip()
        if not part:
            continue
        if ".." in part:
            try:
                start, end = map(int, part.split(".."))
                if 1 <= start <= end <= seq_len:
                    regions.append((start, end))
            except ValueError:
                continue
        else:
            try:
                pos = int(part)
                if 1 <= pos <= seq_len:
                    regions.append((pos, pos))
            except ValueError:
                continue
    return regions

def read_fasta_sequences(fasta_path):
    proteins = {}
    with open(fasta_path, "r") as f:
        acc, seq = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if acc:
                    proteins[acc] = "".join(seq)
                acc = line[1:].strip().split()[0]
                seq = []
            else:
                seq.append(line)
        if acc:
            proteins[acc] = "".join(seq)
    return proteins

def get_processed_accs():
    processed = set()
    for info in FEATURE_MAP.values():
        path = info["output"]
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    processed.update(list(item.keys())[0] for item in data)
            except Exception:
                continue
    return processed

def process_protein(acc, seq):
    try:
        for attempt in range(5):
            try:
                r = requests.get(MOBIDB_TSV_URL.format(acc), timeout=15)
                if r.status_code == 200:
                    break
                time.sleep(2)
            except requests.RequestException:
                time.sleep(2)
        else:
            return None

        label_dict = {key: [0] * len(seq) for key in FEATURE_MAP}
        has_data = False

        for line in r.text.splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            feature, region_str = parts[1], parts[2]
            regions = parse_region(region_str, len(seq))
            if not regions:
                continue

            for key, info in FEATURE_MAP.items():
                if feature in info["pos"]:
                    for start, end in regions:
                        if key == "fold" and (end - start + 1) < 10:
                            continue
                        for i in range(start - 1, end):
                            label_dict[key][i] = 1
                    has_data = True
                elif feature in info["neg"]:
                    for start, end in regions:
                        for i in range(start - 1, end):
                            if label_dict[key][i] == 0:
                                label_dict[key][i] = -1
                    has_data = True

        if not has_data:
            return None

        return acc, label_dict
    except Exception:
        return None

def writer_thread(output_path, queue, total_expected):
    temp_path = output_path + ".tmp"
    buffer = []
    existing_data = []
    total_written = 0
    new_since_last_flush = 0
    flush_count = 0

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        try:
            with open(output_path, "r") as f:
                existing_data = json.load(f)
                total_written = len(existing_data)
                print(f"[Writer] {output_path}: Loaded {total_written} previously written sequences.")
        except Exception as e:
            print(f"[Writer] Warning: Could not load existing data from {output_path}: {e}")

    try:
        while True:
            try:
                item = queue.get(timeout=1)
                if item is None:
                    break
                acc, labels = item
                buffer.append({acc: labels})
                new_since_last_flush += 1
                total_written += 1

                if new_since_last_flush >= WRITE_CHUNK_SIZE:
                    existing_data.extend(buffer)
                    with open(temp_path, "w") as f:
                        json.dump(existing_data, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(temp_path, output_path)

                    flush_count += 1
                    print(f"[Writer] {output_path}: Flushed {new_since_last_flush} new sequences (flush #{flush_count}). Total written: {total_written}")
                    buffer.clear()
                    new_since_last_flush = 0

            except Empty:
                time.sleep(0.1)

    finally:
        if buffer:
            existing_data.extend(buffer)
            with open(temp_path, "w") as f:
                json.dump(existing_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, output_path)
            print(f"[Writer] Final flush to {output_path}: {len(buffer)} remaining. Total written: {total_written}")

def build_label_data(protein_seqs):
    processed_accs = get_processed_accs()
    unprocessed = {acc: seq for acc, seq in protein_seqs.items() if acc not in processed_accs}
    total = len(unprocessed)

    if not total:
        print("[Info] No new proteins to process.")
        return

    queues = {key: Queue() for key in FEATURE_MAP}
    threads = [Thread(target=writer_thread, args=(info["output"], queues[key], total)) for key, info in FEATURE_MAP.items()]
    for t in threads:
        t.start()

    shutdown_requested = False

    def handle_interrupt(signal_num, frame):
        nonlocal shutdown_requested
        print("\n[Warning] Keyboard interruption received. Attempting to safely shut down...")
        shutdown_requested = True

    signal.signal(signal.SIGINT, handle_interrupt)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_protein, acc, seq): acc for acc, seq in unprocessed.items()}
        try:
            for future in tqdm(futures, desc="Processing proteins"):
                if shutdown_requested:
                    print("[Info] Shutdown flag detected. Cancelling all remaining tasks.")
                    break
                result = future.result()
                if result:
                    acc, label_dict = result
                    for key, labels in label_dict.items():
                        queues[key].put((acc, labels))
        except KeyboardInterrupt:
            print("[Error] KeyboardInterrupt during execution.")
        finally:
            for future in futures:
                future.cancel()

    # Signal all writers to shut down
    for queue in queues.values():
        queue.put(None)
    for t in threads:
        t.join()

    print("[Done] All data written. Program terminated cleanly.")


if __name__ == "__main__":
    protein_seqs = read_fasta_sequences(FASTA_PATH)
    build_label_data(protein_seqs)
