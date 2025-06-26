import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

st.set_page_config(page_title="Handwritten Digit Generator", layout="centered")

st.title("Handwritten Digit Image Generator (GAN)")
st.write(
    "This app uses a Generative Adversarial Network (GAN) trained on MNIST to generate images of handwritten digits."
)

# --- Generator Model Definition ---
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

# --- Load or Initialize Generator ---
@st.cache_resource
def load_generator(weights_path=None):
    generator = make_generator_model()
    if weights_path:
        try:
            generator.load_weights(weights_path)
            st.success("Loaded generator weights.")
        except Exception as e:
            st.warning("Could not load weights. Using untrained generator.")
    return generator

# --- Generate and Display Digits ---
def generate_digits(generator, num_images=16, seed=None):
    if seed is not None:
        np.random.seed(seed)
    noise = tf.random.normal([num_images, 100])
    predictions = generator(noise, training=False)
    images = predictions.numpy()
    return images

# --- Main App ---
generator = load_generator()  # You can specify a weights file if you have one

st.header("Generate Handwritten Digits")
num_images = st.slider("Number of digits to generate", 1, 25, 16)
random_seed = st.number_input("Random seed (optional)", min_value=0, max_value=100000, value=42, step=1)

if st.button("Generate Digits"):
    images = generate_digits(generator, num_images=num_images, seed=random_seed)
    ncols = min(5, num_images)
    nrows = int(np.ceil(num_images / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    axs = np.array(axs).reshape(-1)
    for i in range(num_images):
        axs[i].imshow(images[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        axs[i].axis("off")
    for j in range(num_images, len(axs)):
        axs[j].axis("off")
    st.pyplot(fig)

st.info(
    "Note: This app uses an untrained generator by default. "
    "You can upload your own trained weights (Keras H5 format) below to generate realistic digits."
)

uploaded_weights = st.file_uploader("Upload generator weights (.h5 file)", type=["h5"])
if uploaded_weights:
    with open("generator_weights.h5", "wb") as f:
        f.write(uploaded_weights.read())
    generator = load_generator("generator_weights.h5")
    st.success("Uploaded and loaded your weights! Now generate digits above.")

st.markdown("---")
st.caption("Built with TensorFlow and Streamlit.")
