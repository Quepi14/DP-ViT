
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dp_vit import (
    SignatureBatchSampler,
    adaptive_collate_fn,
    get_patch_signature,
)

# CONFIGURATION

BATCH_SIZE = 16
NUM_WORKERS = 4
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# DATASET

def create_datasets():
    """
    Buat dataset training, validation, dan testing.

    Dataset harus menghasilkan:
        image, patch_size_map, label

    Patch map harus konsisten dengan image
    yang nantinya digunakan oleh collate_fn.
    """

    # TODO:
    # Implementasikan dataset sesuai pipeline penelitian.
    # Placeholder sementara agar script dapat dieksekusi
    # dan tuple assignment di main() valid.
    return [], [], []

# PATCH SIGNATURE

def create_signatures(dataset):
    """
    Menghasilkan signature untuk setiap sampel.
    Signature berisi jumlah patch pada masing-masing
    cabang ukuran 8, 16, dan 32.
    """

    signatures = []

    for index in range(len(dataset)):
        sample = dataset[index]

        image, patch_size_map, label = sample

        signature = get_patch_signature(
            patch_size_map
        )

        signatures.append(signature)

    return signatures

# DATA LOADER

def create_train_loader(train_dataset):

    train_signatures = create_signatures(
        train_dataset
    )

    train_sampler = SignatureBatchSampler(
        signatures=train_signatures,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=adaptive_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, train_sampler

# MAIN

def main():

    train_dataset, val_dataset, test_dataset = (
        create_datasets()
    )

    train_loader, train_sampler = create_train_loader(
        train_dataset
    )

    print(f"Device: {DEVICE}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Training batches: {len(train_loader)}")

    # TODO:
    # 1. Inisialisasi model DP-ViT.
    # 2. Inisialisasi optimizer dan scheduler.
    # 3. Jalankan training loop.
    # 4. Evaluasi validation.
    # 5. Simpan checkpoint terbaik.


if __name__ == "__main__":
    main()
