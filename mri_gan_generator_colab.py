"""
mri_gan_generator_colab.py — DCGAN Synthetic MRI Generator (Google Colab)
==========================================================================
Identical architecture to mri_gan_generator.py but configured for
Google Colab + Google Drive paths.

Setup (run once in a Colab cell):
    from google.colab import drive
    drive.mount('/content/drive')

Then run this script cell-by-cell or as a whole.

Architecture
------------
Generator  : Dense → Reshape(8×8×512) → 3× Conv2DTranspose → Conv2D(tanh)
Discriminator: 4× Conv2D (64→128→256→512) → Dense sigmoid
Latent dim : 128
Output     : 64×64 grayscale  (upsampled to 224×224 for the classifier)

Author : Manan Pal  (B.Tech CSE, KIIT University)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — edit DATASET_PATH to your Google Drive folder
# ---------------------------------------------------------------------------
IMG_SIZE:         int = 64
OUTPUT_IMG_SIZE:  int = 224
BATCH_SIZE:       int = 64
LATENT_DIM:       int = 128
EPOCHS:           int = 400
SYNTHETIC_IMAGES: int = 500
PREVIEW_EVERY:    int = 50

# ← Change this to your actual Drive path
DATASET_PATH: str = "/content/drive/MyDrive/brain_tumor_dataset"
CLASSES: List[str] = ["glioma", "meningioma", "notumor", "pituitary"]

OUTPUT_PATH:     str = "/content/synthetic_dataset"
MODEL_SAVE:      str = "/content/gan_models"
PREVIEW_PATH:    str = "/content/gan_preview"
CHECKPOINT_PATH: str = "/content/gan_checkpoints"

for _d in (OUTPUT_PATH, MODEL_SAVE, PREVIEW_PATH, CHECKPOINT_PATH):
    os.makedirs(_d, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_images(folder: str) -> np.ndarray:
    """Load images from *folder* as normalised [-1, 1] float32 arrays.

    Returns ndarray of shape (N, IMG_SIZE, IMG_SIZE, 1).
    """
    images: list = []
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Class folder not found: {folder}")

    for file in folder_path.iterdir():
        if file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        try:
            img = Image.open(file).convert("L").resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
            images.append(np.expand_dims(arr, axis=-1))
        except Exception as exc:
            logger.warning("Skipping %s: %s", file.name, exc)

    if not images:
        raise ValueError(f"No valid images found in {folder}")

    result = np.array(images, dtype=np.float32)
    logger.info("Loaded %d images from %s", len(result), folder)
    return result


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
def build_generator() -> tf.keras.Model:
    """DCGAN generator: noise vector → 64×64 grayscale image in [-1, 1]."""
    model = tf.keras.Sequential(name="generator")
    model.add(tf.keras.layers.Input(shape=(LATENT_DIM,)))
    model.add(tf.keras.layers.Dense(8 * 8 * 512))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Reshape((8, 8, 512)))

    for filters in (256, 128, 64):
        model.add(tf.keras.layers.Conv2DTranspose(filters, 4, strides=2, padding="same"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(0.2))

    model.add(tf.keras.layers.Conv2D(1, 3, padding="same", activation="tanh"))
    return model


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------
def build_discriminator() -> tf.keras.Model:
    """DCGAN discriminator: image → real/fake probability."""
    model = tf.keras.Sequential(name="discriminator")
    model.add(tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)))

    for filters in (64, 128, 256, 512):
        model.add(tf.keras.layers.Conv2D(filters, 4, strides=2, padding="same"))
        model.add(tf.keras.layers.LeakyReLU(0.2))
        model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    return model


# ---------------------------------------------------------------------------
# Preview grid
# ---------------------------------------------------------------------------
def save_preview(generator: tf.keras.Model, epoch: int, class_name: str) -> None:
    """Save a 4×4 grid of generated samples."""
    noise  = tf.random.normal((16, LATENT_DIM))
    images = generator.predict(noise, verbose=0)

    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for ax, img in zip(axes.flat, images):
        ax.imshow((img[:, :, 0] + 1) / 2, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    plt.suptitle(f"{class_name} — epoch {epoch}", fontsize=8)
    plt.tight_layout()
    save_path = os.path.join(PREVIEW_PATH, f"{class_name}_epoch_{epoch:04d}.png")
    plt.savefig(save_path, dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def restore_checkpoint(
    manager: tf.train.CheckpointManager,
    checkpoint: tf.train.Checkpoint,
) -> int:
    """Restore latest checkpoint; return start epoch (0 if none)."""
    latest = manager.latest_checkpoint
    if latest:
        checkpoint.restore(latest)
        start = int(latest.split("-")[-1])
        logger.info("Restored checkpoint %s (epoch %d)", latest, start)
        return start
    logger.info("No checkpoint found — starting from epoch 0")
    return 0


# ---------------------------------------------------------------------------
# GAN training
# ---------------------------------------------------------------------------
def train_gan(images: np.ndarray, class_name: str) -> tf.keras.Model:
    """Train DCGAN on *images* and return the trained generator."""
    dataset = (
        tf.data.Dataset.from_tensor_slices(images)
        .shuffle(buffer_size=min(2000, len(images)))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    generator     = build_generator()
    discriminator = build_discriminator()

    loss_fn  = tf.keras.losses.BinaryCrossentropy()
    opt_gen  = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    opt_disc = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint = tf.train.Checkpoint(
        epoch=tf.Variable(0),
        generator=generator,
        discriminator=discriminator,
        gen_optimizer=opt_gen,
        disc_optimizer=opt_disc,
    )
    ckpt_dir = os.path.join(CHECKPOINT_PATH, class_name)
    manager  = tf.train.CheckpointManager(checkpoint, ckpt_dir, max_to_keep=3)
    start_epoch = restore_checkpoint(manager, checkpoint)

    for epoch in range(start_epoch, EPOCHS):
        d_loss_sum = g_loss_sum = 0.0
        n_batches  = 0

        for real_batch in dataset:
            bsz   = tf.shape(real_batch)[0]
            noise = tf.random.normal((bsz, LATENT_DIM))
            fake  = generator(noise, training=True)

            real_labels = tf.ones((bsz, 1)) * 0.9   # label smoothing
            fake_labels = tf.zeros((bsz, 1))

            # Discriminator update
            with tf.GradientTape() as d_tape:
                d_loss = (
                    loss_fn(real_labels, discriminator(real_batch, training=True))
                    + loss_fn(fake_labels, discriminator(fake,       training=True))
                ) / 2
            opt_disc.apply_gradients(
                zip(d_tape.gradient(d_loss, discriminator.trainable_variables),
                    discriminator.trainable_variables)
            )

            # Generator update
            noise = tf.random.normal((bsz, LATENT_DIM))
            with tf.GradientTape() as g_tape:
                fake  = generator(noise, training=True)
                g_loss = loss_fn(tf.ones((bsz, 1)), discriminator(fake, training=True))
            opt_gen.apply_gradients(
                zip(g_tape.gradient(g_loss, generator.trainable_variables),
                    generator.trainable_variables)
            )

            d_loss_sum += d_loss.numpy()
            g_loss_sum += g_loss.numpy()
            n_batches  += 1

        logger.info(
            "[%s] Epoch %d/%d  D=%.4f  G=%.4f",
            class_name, epoch + 1, EPOCHS,
            d_loss_sum / max(n_batches, 1),
            g_loss_sum / max(n_batches, 1),
        )

        if (epoch + 1) % PREVIEW_EVERY == 0 or epoch == EPOCHS - 1:
            save_preview(generator, epoch + 1, class_name)

        checkpoint.epoch.assign(epoch + 1)
        manager.save()

    gen_path = os.path.join(MODEL_SAVE, f"{class_name}_generator_final.h5")
    generator.save(gen_path)
    logger.info("Generator saved → %s", gen_path)
    return generator


# ---------------------------------------------------------------------------
# Synthetic image generation
# ---------------------------------------------------------------------------
def generate_synthetic_images(generator: tf.keras.Model, class_name: str) -> None:
    """Generate SYNTHETIC_IMAGES images at OUTPUT_IMG_SIZE and save them."""
    out_folder = os.path.join(OUTPUT_PATH, class_name)
    os.makedirs(out_folder, exist_ok=True)

    noise  = tf.random.normal((SYNTHETIC_IMAGES, LATENT_DIM))
    images = generator.predict(noise, verbose=0)

    for i, img_arr in enumerate(images):
        img_arr  = ((img_arr[:, :, 0] + 1) / 2 * 255).astype(np.uint8)
        pil_img  = Image.fromarray(img_arr).resize((OUTPUT_IMG_SIZE, OUTPUT_IMG_SIZE))
        pil_img.save(os.path.join(out_folder, f"synthetic_{i:04d}.png"))

    logger.info("Saved %d synthetic images → %s", SYNTHETIC_IMAGES, out_folder)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    dataset_root = Path(DATASET_PATH)
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root not found: {DATASET_PATH}\n"
            "Mount your Drive and update DATASET_PATH."
        )

    for cls in CLASSES:
        logger.info("=" * 55)
        logger.info("Processing class : %s", cls)
        logger.info("=" * 55)

        images    = load_images(str(dataset_root / cls))
        generator = train_gan(images, cls)
        generate_synthetic_images(generator, cls)

    logger.info("Done. All synthetic images saved to: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
