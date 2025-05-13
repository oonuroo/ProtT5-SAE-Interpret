import os
import sys
sys.path.append("/home/onur/Desktop/Project/proj/src")

import pytorch_lightning as pl
import torch
from sae_model import SparseAutoencoder, loss_fn

## sae_module
    # Sae model
    # Optimizer setup
    # Train and validation steps
    # Early stopping avg_mse_loss 

# No Cross Entropy loss calculation (SAE quality metrics without downstream probe) 
# Because our model is encoder architecture, no decoder. 



class SAELightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.layer_to_use = args.layer_to_use
        self.sae_model = SparseAutoencoder(
            d_model=args.d_model,
            d_hidden=args.d_hidden,
            k=args.k,
            auxk=args.auxk,
            batch_size=args.batch_size,
            dead_steps_threshold=args.dead_steps_threshold,
        )
        self.validation_step_outputs = []

    def forward(self, x):
        return self.sae_model(x)

    def training_step(self, batch, batch_idx):
        embeddings = batch["embedding"]  # Expect shape: (B, L, D_MODEL)
        batch_size = embeddings.size(0)

        recons, auxk, num_dead = self(embeddings)
        mse_loss, auxk_loss = loss_fn(embeddings, recons, auxk)
        loss = mse_loss + auxk_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_mse_loss", mse_loss, on_step=True, on_epoch=True, logger=True, batch_size=batch_size)
        self.log("train_auxk_loss", auxk_loss, on_step=True, on_epoch=True, logger=True, batch_size=batch_size)
        self.log("num_dead_neurons", num_dead, on_step=True, on_epoch=True, logger=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        embeddings = batch["embedding"]
        batch_size = embeddings.size(0)

        recons = self.sae_model.forward_val(embeddings)
        mse_loss, auxk_loss = loss_fn(embeddings, recons, None)

        val_metrics = {
            "mse_loss": mse_loss,
        }
        self.validation_step_outputs.append(val_metrics)
        return val_metrics

    def on_validation_epoch_end(self):
        avg_mse_loss = torch.stack([x["mse_loss"] for x in self.validation_step_outputs]).mean()
        self.log("avg_mse_loss", avg_mse_loss, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr)

    def on_after_backward(self):
        self.sae_model.norm_weights()
        self.sae_model.norm_grad()