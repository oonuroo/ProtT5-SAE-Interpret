import os
import requests
import json
import time
from tqdm import tqdm
from threading import Thread
from queue import Queue, Empty  # Import Empty explicitly
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

# Ensure output directories exist
for path in [DISORDER_OUTPUT_JSON, PHASESEP_OUTPUT_JSON, FOLD_BINDING_OUTPUT_JSON, ELM_OUTPUT_JSON, SUBCLASS_POS1_JSON, SUBCLASS_POS2_JSON]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)

def parse_region(region_str):
    regions = []
    for part in region_str.split(","):
        if ".." in part:
            try:
                start, end = map(int, part.split(".."))
                regions.append((start, end))
            except ValueError:
                continue  # Skip invalid regions
        else:
            try:
                pos = int(part)
                regions.append((pos, pos))
            except ValueError:
                continue  # Skip invalid regions
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
                    continue  # Skip invalid regions

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

def writer_thread(output_path, queue, done_flag):
    buffer = {}
    counter = 0

    # Initialize the file with an opening bracket for a JSON array
    with open(output_path, "w") as f:
        f.write("[\n")

    while not done_flag["done"] or not queue.empty():
        try:
            acc, labels = queue.get(timeout=1)
            buffer[acc] = labels
            counter += 1

            # Write to file in chunks
            if counter % WRITE_CHUNK_SIZE == 0:
                with open(output_path, "a") as f:
                    for i, (acc_key, lbl) in enumerate(buffer.items()):
                        json.dump({acc_key: lbl}, f)
                        if i < len(buffer) - 1 or not queue.empty() or not done_flag["done"]:
                            f.write(",\n")
                        else:
                            f.write("\n")
                    f.flush()
                buffer.clear()

        except Empty:  # Correct exception
            time.sleep(0.1)
        except Exception as e:
            print(f"Error writing to {output_path}: {e}")
            time.sleep(0.1)

    # Write remaining data
    if buffer:
        with open(output_path, "a") as f:
            for i, (acc_key, lbl) in enumerate(buffer.items()):
                json.dump({acc_key: lbl}, f)
                if i < len(buffer) - 1:
                    f.write(",\n")
                else:
                    f.write("\n")
            f.flush()

    # Close the JSON array
    with open(output_path, "a") as f:
        f.write("]\n")
        f.flush()

    print(f"[✓] Written {counter} sequences to {output_path}")

def build_label_data(protein_seqs):
    print(f"[INFO] Starting label extraction for {len(protein_seqs)} proteins")

    queues = {
        "disorder": Queue(),
        "phase": Queue(),
        "fold": Queue(),
        "elm": Queue(),
        "subclass1": Queue(),
        "subclass2": Queue()
    }
    done_flags = {key: {"done": False} for key in queues}

    threads = [
        Thread(target=writer_thread, args=(DISORDER_OUTPUT_JSON, queues["disorder"], done_flags["disorder"])),
        Thread(target=writer_thread, args=(PHASESEP_OUTPUT_JSON, queues["phase"], done_flags["phase"])),
        Thread(target=writer_thread, args=(FOLD_BINDING_OUTPUT_JSON, queues["fold"], done_flags["fold"])),
        Thread(target=writer_thread, args=(ELM_OUTPUT_JSON, queues["elm"], done_flags["elm"])),
        Thread(target=writer_thread, args=(SUBCLASS_POS1_JSON, queues["subclass1"], done_flags["subclass1"])),
        Thread(target=writer_thread, args=(SUBCLASS_POS2_JSON, queues["subclass2"], done_flags["subclass2"])),
    ]

    for t in threads:
        t.start()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_protein_all, acc, seq) for acc, seq in protein_seqs.items()]
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

    for flag in done_flags.values():
        flag["done"] = True
    for t in threads:
        t.join()

if __name__ == "__main__":
    protein_seqs = read_fasta_sequences(FASTA_PATH)
    build_label_data(protein_seqs)
