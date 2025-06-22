import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Cargar modelo generador
@st.cache_resource
def cargar_generador():
    return tf.keras.models.load_model("generador_entrenado.keras")

generador = cargar_generador()

# Parámetros
Z_DIM = 100

# Título de la app
st.title("Generador de Dígitos MNIST (Condicional)")

# Selección de dígito a generar
digito = st.slider("Selecciona un dígito (0-9)", 0, 9, 0)

# Botón para generar imágenes
if st.button("Generar imágenes"):
    z = tf.random.normal([5, Z_DIM])
    lbl = tf.constant([[digito]] * 5)
    imagenes = (generador([z, lbl], training=False) + 1) / 2
    imagenes = imagenes.numpy().squeeze()

    st.subheader("Dígitos generados")
    cols = st.columns(5)
    for i in range(5):
        cols[i].image(imagenes[i], width=100, caption=f"Dígito {digito}", use_container_width=False)
