# DP-ViT

DP-ViT is a PyTorch library for building Vision Transformer models that use
ICS-guided adaptive patch sizes. It can also apply structured magnitude-based
pruning to the MLP layers after training.

The library focuses on the model components:

- Image Complexity Score (ICS) calculation
- Adaptive patch-size assignment
- Vision Transformer model
- Optional structured magnitude-based MLP pruning
- Model checkpoint save and load
- Single-image inference

Dataset loading, image cleaning, augmentation, normalization, and training
loops are left to the user.

> **Status:** This project is under development. The API and implementation
> may change before a stable release.

## How it works

DP-ViT uses image complexity to assign a patch size to each image region.
Regions with different complexity can use different patch sizes. The resulting
patches are projected into tokens and passed to a Transformer encoder.

The optional pruning step removes low-magnitude MLP neurons after training.
Pruning is separate from the training loop; this library does not automatically
fine-tune the model after pruning.

## Installation

### Install from a local clone

From the project root, run:

```bash
pip install -e .