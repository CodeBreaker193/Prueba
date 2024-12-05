import streamlit as st

# Título de la aplicación
st.title("¡Hola desde Streamlit!")

# Texto de bienvenida
st.write("Esta es una prueba para verificar que tu aplicación se despliega correctamente en Streamlit Cloud.")

# Interacción simple
name = st.text_input("¿Cómo te llamas?")
if name:
    st.write(f"¡Hola, {name}! 🎉 Bienvenido/a a tu primera app en Streamlit.")
