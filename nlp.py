"""Inferencia SVM y chatbot extraídos del notebook, sin entrenamiento.

generar_respuesta conserva las reglas originales: no comprueba fechas y el
producto visual reemplaza al textual. La coordinación futura podrá mejorarlas.
"""

from functools import lru_cache
from pathlib import Path
import random

import joblib

from catalogo import ALIASES, COLUMNAS_INTENCIONES, RESPUESTAS, buscar_producto
from nlp_preprocessing import normalizar_texto, preparar_texto, tokenizar_y_stem

RUTA_MODELO = Path(__file__).resolve().parent / "artifacts" / "nlp_intenciones.joblib"


@lru_cache(maxsize=1)
def cargar_modelo(ruta=RUTA_MODELO):
    """Carga una vez el pipeline local de confianza (vectorizador y SVM)."""
    return joblib.load(ruta)


def detectar_intencion_ml(mensaje):
    """Preprocesa el mensaje antes de predecir una de las cinco intenciones."""
    return cargar_modelo().predict([preparar_texto(mensaje)])[0]


def detectar_entidades(mensaje):
    """Conserva la búsqueda por subcadena y la primera coincidencia original."""
    texto = normalizar_texto(mensaje)
    for producto, palabras in ALIASES.items():
        for palabra in palabras:
            if palabra in texto:
                return {"producto": producto}
    return {"producto": "no especificado"}


def generar_respuesta(mensaje, imagen=None):
    """Reutiliza el chatbot; imagen es una etiqueta de fruta, no un archivo."""
    intencion = detectar_intencion_ml(mensaje)
    entidades = detectar_entidades(mensaje)
    if imagen is not None:
        entidades["producto"] = imagen

    if intencion == "desconocida":
        respuesta = (
            "No estoy seguro de haber entendido. "
            "Puedes preguntarme por promociones, ejecución o vigencias de productos"
        )
    elif intencion in ("saludo", "despedida"):
        respuesta = random.choice(RESPUESTAS[intencion])
    else:
        fila = buscar_producto(entidades["producto"])
        if fila is None:
            respuesta = "No encontré información"
        else:
            if intencion == "consultar_vigencia":
                entidades["vigencia"] = (
                    f"desde {fila['fecha_inicio']} hasta {fila['fecha_fin']}"
                )
            else:
                for columna in COLUMNAS_INTENCIONES[intencion]:
                    entidades[columna] = fila[columna]
            respuesta = random.choice(RESPUESTAS[intencion]).format(**entidades)
    return {"intencion": intencion, "entidades": entidades, "respuesta": respuesta}
