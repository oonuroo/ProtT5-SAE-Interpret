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

# hyper_grid_search
    # k = topk active neurons
    # d_hidden = hidden layer size
    # lr = learning rate
    # layer = 16 or 24 from prot 
# For each combination starts model and data module; runs and logs  csv logger and wandb logger

# --- Hyperparameter grid ---
ks = [16, 32, 64, 128, 256]
d_hiddens = [2048, 4096, 8192, 16384] 
layers = [16, 24]
lrs = [1e-4, 2e-4, 5e-4]  # Learning rates to try
auxk = 256                  # For convert dead neurons into active          
d_model = 1024  # ProtT5 hidden size

# --- Training constants ---
layer_to_path = {
    16: "/home/onur/Desktop/Project/last-embed/layer16_embeddings.h5",
    24: "/home/onur/Desktop/Project/last-embed/output_layer_embeddings.h5",
}

# data_dir = "/home/onur/Desktop/Project/last-embed/output_layer_embeddings.h5"  # Replace if needed
# data_dir = "/home/onur/Desktop/Project/last-embed/layer16_embeddings.h5"  # Replace if needed

max_epochs = 10
batch_size = 32
num_devices = 1  # Adjust for multi-GPU if available

# --- Output root directory ---
output_root = "results"
os.makedirs(output_root, exist_ok=True)

# --- W&B login (run once)
wandb.login(key="8ac2985125264f7b714a40f18c929a5cc72966e5")

# --- Start grid search ---
for layer, d_hidden, k, lr in itertools.product(layers, d_hiddens, ks, lrs):
    suffix = f"l{layer}_h{d_hidden}_k{k}_lr{lr:.0e}"
    run_dir = os.path.join(output_root, f"layer{layer}", f"dim{d_hidden}", f"k{k}", f"lr{lr:.0e}")

    os.makedirs(run_dir, exist_ok=True)

    print(f"[INFO] Starting run: {suffix}")

    args = Namespace(
        data_dir= layer_to_path[layer],
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
        esm2_weight=None,  # Not needed for ProtT5
        wandb_project="SAE-Grid-Search",  # Not using wandb here
        dead_steps_threshold=2000
    )

    run_name = f"sae_l{layer}_h{d_hidden}_k{k}_lr{lr:.0e}"

   # --- Loggers
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        save_dir=os.path.join(run_dir, "wandb")
    )   
    csv_logger = CSVLogger(save_dir=os.path.join(run_dir, "logs"), name="sae")

   # --- Model & Data  
    model = SAELightningModule(args)
    data_module = SequenceDataModule(args.data_dir, args.batch_size)

    # --- Callbacks
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
        patience=5,
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
    wandb.finish()
    print(f"[DONE] Finished run: {suffix}\n")
