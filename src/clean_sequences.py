import re

INPUT_FILE = "/home/onur/Desktop/Project/mobidb_silver_clustered_40 (Copy 2)"
OUTPUT_FILE = "/home/onur/Desktop/Project/cleaned_mobidb_silver_clustered_40"




def clean_sequence(seq: str) -> str:
    """Replace rare amino acids and insert space between each."""
    return " ".join(list(re.sub(r"[UZOB]", "X", seq)))

def clean_fasta_file(input_path: str, output_path: str):
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        protein_id = None
        sequence_lines = []

        for line in fin:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if protein_id is not None:
                    full_seq = "".join(sequence_lines)
                    cleaned = clean_sequence(full_seq)
                    fout.write(f">{protein_id}\n{cleaned}\n")
                protein_id = line[1:]
                sequence_lines = []
            else:
                sequence_lines.append(line)

        # Write last sequence
        if protein_id is not None:
            full_seq = "".join(sequence_lines)
            cleaned = clean_sequence(full_seq)
            fout.write(f">{protein_id}\n{cleaned}\n")

    print(f"Cleaned sequences saved to: {output_path}")

# Run
clean_fasta_file(INPUT_FILE, OUTPUT_FILE)