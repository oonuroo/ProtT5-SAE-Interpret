import os
import sys
sys.path.append("/home/onur/Desktop/Project/proj/src")

import h5py
import torch
import argparse
from tqdm import tqdm
from sae_model import SparseAutoencoder

# Loads trained SAE, iterates over .h5 file. 
# Run each embedding through SAE in inference mode
# Extracts the latent activations from encoder layer 
# Saves activations as dictionary {protein_id: activation_tensor} to .pt file.

# To interpret what each neuron is detecting
# Cluster sequences
# Find dead neurons
# Feed into classifier or biological feature analysis 


## Instantiates sae with hyperparameters; loads weights from torch ligthining checkpoint
# Returns model ready for inference
def load_model(checkpoint_path, d_model, d_hidden, k, auxk, batch_size, dead_steps_threshold, device):
    model = SparseAutoencoder(
        d_model=d_model,
        d_hidden=d_hidden,
        k=k,
        auxk=auxk,
        batch_size=batch_size,
        dead_steps_threshold=dead_steps_threshold,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in ckpt:
        # lightning checkpoint
        model.load_state_dict({k.replace("sae_model.", ""): v for k, v in ckpt["state_dict"].items()})
    else:
        model.load_state_dict(ckpt)
    model.eval()
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", required=True, help="Path to .h5 embedding file")
    parser.add_argument("--checkpoint", required=True, help="Path to trained SAE checkpoint")
    parser.add_argument("--output", required=True, help="Path to output .pt file")
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--d-hidden", type=int, required=True)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--auxk", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dead-steps-threshold", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for debugging")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(
        args.checkpoint, args.d_model, args.d_hidden, args.k, args.auxk, args.batch_size, args.dead_steps_threshold, device
    )

    activations = {}
    with h5py.File(args.h5_path, "r") as h5f:
        for i, prot_id in enumerate(tqdm(h5f.keys(), desc="Extracting SAE activations")):
            if args.limit and i >= args.limit:
                break
            embedding = torch.tensor(h5f[prot_id][:], dtype=torch.float32).unsqueeze(0).to(device)  # shape: [1, L, D]
            with torch.no_grad():
                acts = model.get_acts(embedding)  # shape: [1, L, D_HIDDEN]
            activations[prot_id] = acts.squeeze(0).cpu()

    torch.save(activations, args.output)
    print(f"[DONE] Saved latent activations to: {args.output}")


if __name__ == "__main__":
    main()