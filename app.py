"""Interfaz local: streamlit run app.py."""

import os

import requests
import streamlit as st

API_URL = os.environ.get(
    "RETAIL_API_URL", f"http://127.0.0.1:{os.environ.get('RETAIL_API_PORT', '5001')}"
).rstrip("/")

st.set_page_config(page_title="Asistente de Ejecución Retail", page_icon="🍊")
st.title("🍊 Asistente de Ejecución Retail")
st.caption("Consulta promociones, ejecución y vigencias de frutas.")
st.markdown(
    "Puedes preguntar: **¿Qué promoción tiene la naranja?**, "
    "**¿Cómo debo exhibir Granny-Smith?** o **¿Cuándo termina la oferta de Satsumas?**"
)
st.caption("Cada consulta es independiente. Para aclarar un conflicto, envía el nombre sin imagen.")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for entrada in st.session_state.mensajes:
    with st.chat_message(entrada["rol"]):
        st.write(entrada["texto"])
        if entrada.get("imagen"):
            st.image(entrada["imagen"], width=240)

with st.form("consulta", clear_on_submit=True):
    mensaje = st.text_input("¿Qué necesitas saber?")
    archivo = st.file_uploader("Adjunta una foto del producto (opcional)", type=["jpg", "jpeg", "png", "webp"])
    st.caption("Si envías solo una imagen, consultaré su promoción. Máximo 10 MB por solicitud.")
    enviar = st.form_submit_button("Enviar")

if enviar:
    contenido = archivo.getvalue() if archivo is not None else None
    if not mensaje.strip() and not contenido:
        st.warning("Escribe una pregunta o adjunta una imagen.")
    elif contenido and len(contenido) >= 10 * 1024 * 1024:
        st.error("La imagen debe ocupar menos de 10 MB.")
    else:
        try:
            with st.spinner("Consultando los modelos locales…"):
                if contenido:
                    resultado = requests.post(
                        f"{API_URL}/consultar", data={"mensaje": mensaje},
                        files={"imagen": (archivo.name, contenido, archivo.type)}, timeout=120,
                    )
                else:
                    resultado = requests.post(
                        f"{API_URL}/consultar", json={"mensaje": mensaje}, timeout=120,
                    )
                try:
                    datos = resultado.json()
                except ValueError:
                    st.error(
                        f"El servidor en {API_URL}/consultar devolvió HTTP {resultado.status_code} "
                        "sin JSON válido. Comprueba que Flask esté ejecutándose en esa dirección. "
                        "En macOS, AirPlay puede ocupar el puerto 5000; usa el puerto 5001."
                    )
                    datos = None
                if datos is None:
                    pass
                elif not isinstance(datos, dict):
                    st.error("El servidor devolvió un formato inesperado: se esperaba un objeto JSON.")
                elif not resultado.ok:
                    st.error(datos.get("error", "El servidor rechazó la consulta."))
                elif not isinstance(datos.get("respuesta"), str):
                    st.error("La respuesta del servidor no contiene el campo de texto 'respuesta'.")
                else:
                    st.session_state.mensajes.extend([
                        {"rol": "user", "texto": mensaje or "Consultar promoción de la foto", "imagen": contenido},
                        {"rol": "assistant", "texto": datos["respuesta"]},
                    ])
                    st.rerun()
        except requests.Timeout:
            st.error("La consulta tardó demasiado. Intenta nuevamente; la primera carga puede tardar.")
        except requests.ConnectionError:
            st.error("No pude conectar con Flask. Inicia el servidor con python api.py.")
        except requests.RequestException as error:
            st.error(f"Falló la solicitud a {API_URL}: {error}")
