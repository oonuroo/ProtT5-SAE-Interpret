import os
import requests
import json
import time
from tqdm import tqdm
from threading import Thread
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

FASTA_PATH = "/home/onur/Desktop/Project/data/mobidb_silver_clustered_40"
DISORDER_OUTPUT_JSON = "/home/onur/Desktop/Project/proj/databases/tsv/disorder_regions/disorder_labels.json"
PHASESEP_OUTPUT_JSON = "/home/onur/Desktop/Project/proj/databases/tsv/phase_seperation/phase_sep_labels.json"
FOLD_BINDING_OUTPUT_JSON = "/home/onur/Desktop/Project/proj/databases/tsv/fold_upon_binding/fold_binding_labels.json"
ELM_OUTPUT_JSON = "/home/onur/Desktop/Project/proj/databases/tsv/elm/elm_labels.json"
SUBCLASS_POS1_JSON = "/home/onur/Desktop/Project/proj/databases/tsv/sub-classes/pos_1/subclass_pos1.json"
SUBCLASS_POS2_JSON = "/home/onur/Desktop/Project/proj/databases/tsv/sub-classes/pos_2/subclass_pos2.json"

MOBIDB_TSV_URL = "https://mobidb.org/api/download?acc={}&format=tsv"

DISORDER_POS = {"curated-disorder-disprot", "homology-disorder-disprot"}
DISORDER_NEG = {"derived-observed-th_90"}
PHASE_POS = {"curated-phase_separation-merge", "homology-phase_separation-merge"}
FOLD_POS = {"curated-lip-dibs", "homology-lip-dibs", "prediction-lip-alphafold"}
ELM_POS = {"curated-lip-elm", "homology-lip-elm"}
SUBCLASS_POS1 = {"prediction-compact-mobidb_lite_sub"}
SUBCLASS_POS2 = {"prediction-extended-mobidb_lite_sub"}

WRITE_CHUNK_SIZE = 200
NUM_WORKERS = 4

OUTPUT_PATHS = {
    "disorder": DISORDER_OUTPUT_JSON,
    "phase": PHASESEP_OUTPUT_JSON,
    "fold": FOLD_BINDING_OUTPUT_JSON,
    "elm": ELM_OUTPUT_JSON,
    "subclass1": SUBCLASS_POS1_JSON,
    "subclass2": SUBCLASS_POS2_JSON,
}

def parse_region(region_str):
    regions = []
    for part in region_str.split(","):
        if ".." in part:
            try:
                start, end = map(int, part.split(".."))
                regions.append((start, end))
            except ValueError:
                continue
        else:
            try:
                pos = int(part)
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
    """Read accession IDs from all output JSON files and return their intersection."""
    all_accs = []
    for output_path in OUTPUT_PATHS.values():
        accs = set()
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            try:
                with open(output_path, "r") as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        print(f"Error: {output_path} does not contain a JSON list")
                        continue
                    for item in data:
                        if not isinstance(item, dict) or len(item) != 1:
                            print(f"Warning: Invalid item in {output_path}: {item}")
                            continue
                        acc = list(item.keys())[0]
                        accs.add(acc)
                    print(f"[DEBUG] Found {len(accs)} accession IDs in {output_path}")
            except json.JSONDecodeError as e:
                print(f"JSON decode error in {output_path}: {e}")
            except Exception as e:
                print(f"Error reading {output_path}: {e}")
        all_accs.append(accs)
    if not all_accs:
        return set()
    return set.intersection(*all_accs)

def process_protein_all(acc, seq):
    try:
        url = MOBIDB_TSV_URL.format(acc)
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            print(f"Failed to fetch data for {acc}: HTTP {r.status_code}")
            return None

        disorder_labels = [0] * len(seq)
        phase_labels = [0] * len(seq)
        fold_labels = [0] * len(seq)
        elm_labels = [0] * len(seq)
        subclass1_labels = [0] * len(seq)
        subclass2_labels = [0] * len(seq)

        for line in r.text.splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            feature, region_str = parts[1], parts[2]

            for start, end in parse_region(region_str):
                if end > len(seq) or start < 1:
                    continue

                if feature in DISORDER_POS:
                    for i in range(start, end + 1):
                        disorder_labels[i - 1] = 1
                elif feature in DISORDER_NEG:
                    for i in range(start, end + 1):
                        if disorder_labels[i - 1] == 0:
                            disorder_labels[i - 1] = -1
                elif feature in PHASE_POS:
                    for i in range(start, end + 1):
                        phase_labels[i - 1] = 1
                elif feature in FOLD_POS and (end - start + 1) >= 10:
                    for i in range(start, end + 1):
                        fold_labels[i - 1] = 1
                elif feature in ELM_POS:
                    for i in range(start, end + 1):
                        elm_labels[i - 1] = 1
                elif feature in SUBCLASS_POS1:
                    for i in range(start, end + 1):
                        subclass1_labels[i - 1] = 1
                elif feature in SUBCLASS_POS2:
                    for i in range(start, end + 1):
                        subclass2_labels[i - 1] = 1

        return acc, disorder_labels, phase_labels, fold_labels, elm_labels, subclass1_labels, subclass2_labels
    except Exception as e:
        print(f"Error processing {acc}: {e}")
        return None
    
def writer_thread(output_path, queue):
    # Load existing records to support resume
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        try:
            with open(output_path, "r") as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = []
    else:
        existing_data = []

    buffer = []
    counter = 0

    while True:
        try:
            item = queue.get(timeout=1)
            if item is None:
                break

            acc, labels = item
            buffer.append({acc: labels})
            counter += 1

            if len(buffer) >= WRITE_CHUNK_SIZE:
                flush_buffer(buffer, existing_data, output_path)
                buffer.clear()

        except Empty:
            time.sleep(0.1)
        except Exception as e:
            print(f"Error writing to {output_path}: {e}")
            time.sleep(0.1)

    if buffer:
        flush_buffer(buffer, existing_data, output_path)

    print(f"[✓] Written {counter} sequences to {output_path}")


def flush_buffer(buffer, existing_data, output_path):
    existing_data.extend(buffer)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(existing_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)


def build_label_data(protein_seqs):
    # Get set of already processed accession IDs across all output files
    processed_accs = get_processed_accs()
    print(f"[INFO] Found {len(processed_accs)} already processed proteins")

    # Filter unprocessed proteins
    unprocessed_seqs = {acc: seq for acc, seq in protein_seqs.items() if acc not in processed_accs}
    print(f"[INFO] Starting label extraction for {len(unprocessed_seqs)} unprocessed proteins")

    if not unprocessed_seqs:
        print("[INFO] No new proteins to process")
        return

    queues = {key: Queue() for key in OUTPUT_PATHS}

    threads = [
        Thread(target=writer_thread, args=(OUTPUT_PATHS["disorder"], queues["disorder"])),
        Thread(target=writer_thread, args=(OUTPUT_PATHS["phase"], queues["phase"])),
        Thread(target=writer_thread, args=(OUTPUT_PATHS["fold"], queues["fold"])),
        Thread(target=writer_thread, args=(OUTPUT_PATHS["elm"], queues["elm"])),
        Thread(target=writer_thread, args=(OUTPUT_PATHS["subclass1"], queues["subclass1"])),
        Thread(target=writer_thread, args=(OUTPUT_PATHS["subclass2"], queues["subclass2"])),
    ]

    for t in threads:
        t.start()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_protein_all, acc, seq) for acc, seq in unprocessed_seqs.items()]
        for future in tqdm(futures, desc="Processing proteins"):
            result = future.result()
            if result:
                acc, d, p, f, e, s1, s2 = result
                queues["disorder"].put((acc, d))
                queues["phase"].put((acc, p))
                queues["fold"].put((acc, f))
                queues["elm"].put((acc, e))
                queues["subclass1"].put((acc, s1))
                queues["subclass2"].put((acc, s2))

    # Send sentinel values to all queues to stop writer threads
    for queue in queues.values():
        queue.put(None)
    for t in threads:
        t.join()

if __name__ == "__main__":
    protein_seqs = read_fasta_sequences(FASTA_PATH)
    build_label_data(protein_seqs)