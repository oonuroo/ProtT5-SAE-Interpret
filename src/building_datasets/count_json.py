import json

# Replace with the actual path to your file
file_path = '/home/onur/Desktop/Project/proj/databases/tsv/sub-classes/pos_2/subclass_pos2.json'

with open(file_path, 'r') as f:
    data = json.load(f)

entry_count = len(data)
print(f"Number of entries: {entry_count}")
