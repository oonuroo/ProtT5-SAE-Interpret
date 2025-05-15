import os
import json
import requests
import threading
from time import sleep
from Bio import SeqIO
from tqdm import tqdm
from queue import Queue, Empty

# === Config ===
FASTA_PATH = "/home/onur/Desktop/Project/data/mobidb_silver_clustered_40"
OUTPUT_JSON = "/home/onur/Desktop/Project/proj/databases/json/subcellular_localization/subcellular_localizations.json"
FAILED_LOG = "/home/onur/Desktop/Project/proj/databases/json/subcellular_localization/failed_ids.txt"
API_URL = "https://mobidb.org/api/download?acc={}&format=json&projection=acc,localization"
MAX_RETRIES = 5
BATCH_WRITE_SIZE = 250
QUEUE_SIZE = 500

# === Safe JSON loading ===
def load_json_safe(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r") as f:
            return json.load(f)
    return {}

results = load_json_safe(OUTPUT_JSON)

# === Load failed IDs ===
failed_ids = set()
if os.path.exists(FAILED_LOG):
    with open(FAILED_LOG, "r") as f:
        failed_ids.update([line.strip() for line in f.readlines()])

# === Extract protein IDs from FASTA ===
print(" Loading protein IDs from FASTA...")
all_ids = set()
for record in SeqIO.parse(FASTA_PATH, "fasta"):
    uniprot_id = record.id.split("|")[1] if "|" in record.id else record.id
    all_ids.add(uniprot_id)

pending_ids = sorted(all_ids - set(results.keys()) - failed_ids)
print(f" Total to process: {len(pending_ids)} proteins.")

# === Shared resources ===
id_queue = Queue(maxsize=QUEUE_SIZE)
lock = threading.Lock()
buffer = {}
progress = tqdm(total=len(pending_ids), desc="Processing proteins")

# === Write buffer to JSON ===
def flush_buffer_to_disk():
    global buffer
    if not buffer:
        return
    existing = load_json_safe(OUTPUT_JSON)
    existing.update(buffer)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(existing, f, indent=2)
    buffer.clear()

# === API Worker ===
def api_worker():
    global buffer
    while True:
        try:
            protein_id = id_queue.get(timeout=5)
        except Empty:
            break

        url = API_URL.format(protein_id)
        success = False

        for attempt in range(MAX_RETRIES):
            try:
                res = requests.get(url, timeout=10)
                if res.status_code == 200:
                    json_data = res.json()
                    loc = json_data.get("localization", None)
                    with lock:
                        buffer[protein_id] = loc  # None is fine
                    success = True
                    break
                else:
                    sleep(1)
            except Exception:
                sleep(2)

        if not success:
            with open(FAILED_LOG, "a") as f:
                f.write(protein_id + "\n")

        progress.update(1)

        with lock:
            if len(buffer) >= BATCH_WRITE_SIZE:
                flush_buffer_to_disk()

        id_queue.task_done()

# === Launch Threads ===
threads = []
n_threads = 4
for _ in range(n_threads):
    t = threading.Thread(target=api_worker)
    t.start()
    threads.append(t)

# === Feed Queue ===
for pid in pending_ids:
    id_queue.put(pid)

# === Wait for threads ===
for t in threads:
    t.join()

# === Final flush ===
flush_buffer_to_disk()
progress.close()
print(" Done! Remaining buffer flushed.")
