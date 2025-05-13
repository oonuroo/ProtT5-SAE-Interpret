import sys
sys.path.append("/home/onur/Desktop/Project/proj/src")

import os
import itertools
from argparse import Namespace
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import wandb

from sae.data_module import H5DataModule as SequenceDataModule
from sae.sae_module import SAELightningModule


# test_hyper_grid_search
    # test runner for hyper_grid_search with 2.000 sequences


# --- Hyperparameter grid (small test subset) ---
ks = [64]
d_hiddens = [4096]
layers = [16]
lrs = [2e-4]
auxk = 256
d_model = 1024

# --- Training constants ---
data_dir = "/home/onur/Desktop/Project/last-embed/output_layer_embeddings.h5"
max_epochs = 2
batch_size = 32
num_devices = 1

# --- Output directory ---
output_root = "test_results"
os.makedirs(output_root, exist_ok=True)

# --- W&B login ---
wandb.login(key="8ac2985125264f7b714a40f18c929a5cc72966e5")

# --- Run test grid ---
for layer, d_hidden, k, lr in itertools.product(layers, d_hiddens, ks, lrs):
    suffix = f"test_l{layer}_h{d_hidden}_k{k}_lr{lr:.0e}"
    run_dir = os.path.join(output_root, f"layer{layer}", f"dim{d_hidden}", f"k{k}", f"lr{lr:.0e}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"[INFO] Starting test run: {suffix}")

    args = Namespace(
        data_dir=data_dir,
        layer_to_use=layer,
        d_model=d_model,
        d_hidden=d_hidden,
        k=k,
        auxk=auxk,
        batch_size=batch_size,
        lr=lr,
        max_epochs=max_epochs,
        num_devices=num_devices,
        model_suffix=suffix,
        output_dir=run_dir,
        num_workers=None,
        esm2_weight=None,
        wandb_project="SAE-Grid-Search-Test",
        dead_steps_threshold=2000 
    )

    run_name = f"TEST_sae_l{layer}_h{d_hidden}_k{k}_lr{lr:.0e}"

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        save_dir=os.path.join(run_dir, "wandb")
    )
    csv_logger = CSVLogger(save_dir=os.path.join(run_dir, "logs"), name="sae")

    model = SAELightningModule(args)
    data_module = SequenceDataModule(args.data_dir, args.batch_size, limit=2000)  # Only 2000 samples

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename=f"sae-{suffix}-{{step}}-{{avg_mse_loss:.2f}}",
        save_top_k=1,
        monitor="avg_mse_loss",
        mode="min",
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="avg_mse_loss",
        patience=2,
        verbose=True,
        mode="min"
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.num_devices,
        logger=[csv_logger, wandb_logger],
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=1.0,
        val_check_interval=100,
        limit_val_batches=10
    )

    trainer.fit(model, data_module)
    print(f"[DONE] Finished test run: {suffix}\n")
