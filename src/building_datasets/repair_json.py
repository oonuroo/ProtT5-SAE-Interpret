import os
import json

def fix_json_file(path):
    print(f"🔧 Fixing: {path}")
    with open(path, "r") as f:
        content = f.read().strip()

    # Try to load — if valid, skip
    try:
        json.loads(content)
        print(f"✅ Already valid: {path}")
        return
    except json.JSONDecodeError:
        pass

    # Try to fix common format error: remove trailing comma
    if content.endswith(','):
        content = content[:-1]

    if content.endswith(',\n]'):
        content = content.replace(',\n]', '\n]')

    # If it was written like {entry}, {entry}, ... } — fix it into a list
    if content.startswith("{") and content.endswith("}"):
        lines = content.splitlines()
        items = []
        for line in lines:
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                try:
                    items.append(json.loads("{" + line + "}"))
                except:
                    continue
        with open(path, "w") as f:
            json.dump(items, f, indent=2)
        print(f"✅ Fixed object-to-list: {path}")
        return

    # Attempt to wrap in array if needed
    if not content.startswith("["):
        content = "[" + content
    if not content.endswith("]"):
        content = content.rstrip(",") + "]"

    # Save fixed content
    try:
        data = json.loads(content)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Fixed and saved: {path}")
    except Exception as e:
        print(f"❌ Failed to fix {path}: {e}")

# === List your label paths ===
paths = [
    "/home/onur/Desktop/Project/proj/databases/tsv/disorder_regions/disorder_labels.json",
    "/home/onur/Desktop/Project/proj/databases/tsv/phase_seperation/phase_sep_labels.json",
    "/home/onur/Desktop/Project/proj/databases/tsv/fold_upon_binding/fold_binding_labels.json",
    "/home/onur/Desktop/Project/proj/databases/tsv/elm/elm_labels.json",
    "/home/onur/Desktop/Project/proj/databases/tsv/sub-classes/pos_1/subclass_pos1.json",
    "/home/onur/Desktop/Project/proj/databases/tsv/sub-classes/pos_2/subclass_pos2.json"
]

for path in paths:
    if os.path.exists(path):
        fix_json_file(path)
    else:
        print(f"⚠️ File not found: {path}")
