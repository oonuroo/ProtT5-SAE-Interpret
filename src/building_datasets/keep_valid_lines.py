import json

bad_path = "/home/onur/Desktop/Project/proj/databases/tsv/disorder_regions/disorder_labels.json"
fixed_path = bad_path.replace(".json", "_fixed.json")

fixed = []
with open(bad_path, "r") as f:
    try:
        data = json.load(f)
        print("✅ Already valid — no need to fix")
        exit()
    except json.JSONDecodeError:
        pass

    # Try line by line recovery
    f.seek(0)
    for i, line in enumerate(f):
        line = line.strip().rstrip(",")
        if not line or line in {"[", "]", "}"}:
            continue
        try:
            entry = json.loads(line)
            fixed.append(entry)
        except Exception:
            print(f"⚠️ Skipping line {i + 1} due to parse error")

# Write valid entries
with open(fixed_path, "w") as f:
    json.dump(fixed, f, indent=2)

print(f"✅ Recovered and saved to: {fixed_path}")
