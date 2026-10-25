"""
================================================================================
SANKALAN AI PLATFORM - VAE MODEL MODULE
File: backend/models/vae.py
Notice: The main Variational Autoencoder implementation is located in:
        backend/models/main_vae_model.py
================================================================================
This file re-exports all components from main_vae_model.py for seamless compatibility.
"""

from .main_vae_model import (
    VAE,
    Sampling,
    compute_reconstruction_loss,
    get_beta,
    train_step,
    train_vae,
)

__all__ = [
    "VAE",
    "Sampling",
    "compute_reconstruction_loss",
    "get_beta",
    "train_step",
    "train_vae",
]
