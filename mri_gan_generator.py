import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# ======================
# SETTINGS
# ======================

IMG_SIZE = 64
BATCH_SIZE = 64
LATENT_DIM = 128
EPOCHS = 400
SYNTHETIC_IMAGES = 500

DATASET_PATH = r"C:\AD Project\brain_tumor_detection\Training"

CLASSES = ["glioma","meningioma","pituitary","notumor"]

OUTPUT_PATH = "synthetic_dataset"
MODEL_SAVE = "gan_models"
PREVIEW_PATH = "gan_preview"
CHECKPOINT_PATH = "gan_checkpoints"

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE, exist_ok=True)
os.makedirs(PREVIEW_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# ======================
# LOAD IMAGES
# ======================

def load_images(folder):

    images = []

    for file in os.listdir(folder):

        path = os.path.join(folder,file)

        try:

            img = Image.open(path).convert("L")
            img = img.resize((IMG_SIZE,IMG_SIZE))

            img = np.array(img)/127.5 - 1
            img = np.expand_dims(img,-1)

            images.append(img)

        except:
            pass

    return np.array(images)

# ======================
# GENERATOR
# ======================

def build_generator():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(8*8*512, input_dim=LATENT_DIM))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((8,8,512)))

    model.add(tf.keras.layers.Conv2DTranspose(256,4,strides=2,padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(128,4,strides=2,padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64,4,strides=2,padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2D(1,3,padding="same",activation="tanh"))

    return model

# ======================
# DISCRIMINATOR
# ======================

def build_discriminator():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(64,4,strides=2,padding="same",
             input_shape=(IMG_SIZE,IMG_SIZE,1)))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.Conv2D(128,4,strides=2,padding="same"))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.Conv2D(256,4,strides=2,padding="same"))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.Conv2D(512,4,strides=2,padding="same"))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

    return model

# ======================
# SAVE PREVIEW
# ======================

def save_preview(generator,epoch,class_name):

    noise = tf.random.normal((16,LATENT_DIM))
    images = generator.predict(noise,verbose=0)

    fig = plt.figure(figsize=(4,4))

    for i in range(16):

        plt.subplot(4,4,i+1)

        img = (images[i]+1)/2
        plt.imshow(img[:,:,0],cmap="gray")
        plt.axis("off")

    path = f"{PREVIEW_PATH}/{class_name}_epoch_{epoch}.png"
    plt.savefig(path)
    plt.close()

# ======================
# LOAD CHECKPOINT
# ======================

def load_checkpoint(manager, checkpoint):

    latest = manager.latest_checkpoint

    if latest:

        print("Restoring from:", latest)

        checkpoint.restore(latest)

        start_epoch = int(latest.split("-")[-1])

        return start_epoch

    else:

        print("Starting training from scratch")

        return 0

# ======================
# TRAIN GAN
# ======================

def train_gan(images,class_name):

    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.shuffle(2000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    generator = build_generator()
    discriminator = build_discriminator()

    loss_fn = tf.keras.losses.BinaryCrossentropy()

    opt_gen = tf.keras.optimizers.Adam(0.0002,0.5)
    opt_disc = tf.keras.optimizers.Adam(0.0002,0.5)

    checkpoint = tf.train.Checkpoint(
        epoch=tf.Variable(0),
        generator=generator,
        discriminator=discriminator,
        gen_optimizer=opt_gen,
        disc_optimizer=opt_disc
    )

    manager = tf.train.CheckpointManager(
        checkpoint,
        os.path.join(CHECKPOINT_PATH,class_name),
        max_to_keep=5
    )

    start_epoch = load_checkpoint(manager, checkpoint)

    for epoch in range(start_epoch, EPOCHS):

        for real in dataset:

            batch_size = real.shape[0]

            noise = tf.random.normal((batch_size,LATENT_DIM))
            fake = generator(noise)

            real_labels = tf.ones((batch_size,1))*0.9
            fake_labels = tf.zeros((batch_size,1))

            with tf.GradientTape() as disc_tape:

                pred_real = discriminator(real)
                loss_real = loss_fn(real_labels,pred_real)

                pred_fake = discriminator(fake)
                loss_fake = loss_fn(fake_labels,pred_fake)

                loss_disc = (loss_real + loss_fake)/2

            grads = disc_tape.gradient(loss_disc,
                                       discriminator.trainable_variables)

            opt_disc.apply_gradients(
                zip(grads,discriminator.trainable_variables)
            )

            noise = tf.random.normal((batch_size,LATENT_DIM))

            with tf.GradientTape() as gen_tape:

                fake = generator(noise)
                pred = discriminator(fake)

                loss_gen = loss_fn(tf.ones((batch_size,1)),pred)

            grads = gen_tape.gradient(loss_gen,
                                      generator.trainable_variables)

            opt_gen.apply_gradients(
                zip(grads,generator.trainable_variables)
            )

        print(f"Epoch {epoch}/{EPOCHS}  Generator Loss: {loss_gen.numpy():.4f}")

        save_preview(generator,epoch,class_name)

        checkpoint.epoch.assign(epoch)
        manager.save()

    generator.save(f"{MODEL_SAVE}/{class_name}_generator_final.h5")

    return generator

# ======================
# GENERATE IMAGES
# ======================

def generate_images(generator,class_name):

    folder = os.path.join(OUTPUT_PATH,class_name)
    os.makedirs(folder,exist_ok=True)

    noise = tf.random.normal((SYNTHETIC_IMAGES,LATENT_DIM))
    images = generator.predict(noise,verbose=0)

    for i,img in enumerate(images):

        img = (img+1)/2
        img = (img*255).astype(np.uint8)

        img = img[:,:,0]
        img = Image.fromarray(img)

        img = img.resize((224,224))

        img.save(f"{folder}/synthetic_{i}.png")

# ======================
# MAIN LOOP
# ======================

for cls in CLASSES:

    print("\n=======================")
    print("Processing:",cls)
    print("=======================\n")

    path = os.path.join(DATASET_PATH,cls)

    images = load_images(path)

    print("Images loaded:",len(images))

    generator = train_gan(images,cls)

    print("Generating synthetic images...")

    generate_images(generator,cls)

    print("Finished:",cls)

print("\nAll classes completed successfully.")