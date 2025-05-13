
import argparse
import glob
import os

import pytorch_lightning as pl
import wandb
# from data_module import H5DataModule
from data_module import H5DataModule as ProtT5H5DataModule
from sae_module import SAELightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split

import torch
import numpy as np
import random


# ----------------------------
# Reproducibility
# ----------------------------

seed = 42
pl.seed_everything(seed, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)

# ----------------------------
# Argument Parser
# ----------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--data-dir", type=str, required=True, help="Path to HDF5 file with ProtT5 embeddings")
parser.add_argument("--layer-to_use", type=int, default=24, help="Layer number (16 or 24)")
parser.add_argument("--d-hidden", type=int, default=8192, help="Hidden dim of SAE")
parser.add_argument("--k", type=int, default=256, help="Sparsity")
parser.add_argument("--auxk", type=int, default=256, help="Auxiliary k")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--batch-size", type=int, default=1)                                   # We are using batch size 1 because of different length sequences
parser.add_argument("--max-epochs", type=int, default=30)
parser.add_argument("--num-devices", type=int, default=1)
parser.add_argument("--model-suffix", type=str, default="final_train_fullset")
parser.add_argument("--wandb-project", type=str, default="protT5_final_sae")
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--d-model", type=int, default=1024, help="Embedding dimension of input vectors")
parser.add_argument("--dead-steps-threshold", type=int, default=2000, help="Steps before pruning dead neurons")



args = parser.parse_args()

# ----------------------------
# Output Directory Setup
# ----------------------------
args.output_dir = (
    f"/home/onur/Desktop/Project/proj/activations/sae_l{args.layer_to_use}_h{args.d_hidden}_k{args.k}_lr{args.lr}_{args.model_suffix}"
)

os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

sae_name = (
    f"sae_l{args.layer_to_use}_h{args.d_hidden}_k{args.k}_lr{args.lr}_{args.model_suffix}"
)


early_stop_callback = EarlyStopping(
    monitor="train_loss",     # You're not using val_loss, so train_loss makes sense
    min_delta=0.0001,         # Minimum improvement to qualify as an actual improvement
    patience=3,               # If no improvement for 3 evaluations, stop
    verbose=True,
    mode="min"
)

# ----------------------------
# WandB Logger
# ----------------------------

wandb_logger = WandbLogger(
    project=args.wandb_project,
    name=sae_name,
    save_dir=os.path.join(args.output_dir, "wandb"),
)

wandb_logger.log_hyperparams(vars(args))

# ----------------------------
# Data Module and Model + (train set +val set) merge 
# ----------------------------

data_module = ProtT5H5DataModule(args.data_dir, args.batch_size, args.num_workers)

## Merging training set + validation set since we already found best parameters; we do not need validation set 

data_module.setup()

# Split into 90% train, 10% test
data_module.train_ids, data_module.test_ids = train_test_split(
    data_module.all_ids, test_size=0.15, random_state=42
)

# Disable validation completely
data_module.val_ids = []

# Merge train + val IDs for full training set
assert isinstance(data_module.train_ids, list) and isinstance(data_module.val_ids, list), \
    "train_ids or val_ids are not lists"

assert len(data_module.test_ids) > 0, "Test set is empty!"

print(f"[INFO] Total proteins loaded: {len(data_module.all_ids)}")


# Now log true counts
wandb_logger.experiment.config.update({
    "n_train_proteins": len(data_module.train_ids),
    "n_test_proteins": len(data_module.test_ids)
})

print(f"[INFO] Training with {len(data_module.train_ids)} proteins, testing on {len(data_module.test_ids)}.")

# ----------------------------
# Model Init
# ----------------------------

model = SAELightningModule(args)

# ----------------------------
# Checkpoint Callback 
# ----------------------------

checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(args.output_dir, "checkpoints"),
    filename=sae_name + "-{step}-{train_loss:.4f}",
    save_top_k=10,
    monitor="train_loss",
    mode="min",
    save_last=True,
)

# ----------------------------
# Trainer
# ----------------------------
trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    accelerator="gpu",
    devices=list(range(args.num_devices)),
    strategy="auto",
    logger=wandb_logger,
    log_every_n_steps=10,
    val_check_interval=100,
    limit_val_batches=0.0,  # disable validation
    limit_train_batches=1.0,
    callbacks=[checkpoint_callback,early_stop_callback],
    gradient_clip_val=1.0,
)

# ----------------------------
# Training + Logging Artifacts
# ----------------------------

trainer.fit(model, data_module)
trainer.test(model, datamodule=data_module)

# Upload best checkpoints to WandB
for checkpoint in glob.glob(os.path.join(args.output_dir, "checkpoints", "*.ckpt")):
    wandb.log_artifact(checkpoint, type="model")

wandb.finish()