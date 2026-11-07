"""Coordinación del asistente de ejecución comercial."""

import nlp
import vision


def resolver_producto(
    producto_texto: str | None, producto_imagen: str | None
) -> dict[str, str | None]:
    """Decide qué producto usar a partir del texto y de la imagen.

    Entradas:
        producto_texto: nombre canónico detectado en el mensaje, o None.
            El detector NLP devuelve "no especificado" cuando no encuentra
            producto; aquí también debe interpretarse como ausencia.
        producto_imagen: nombre canónico predicho por visión, o None si no
            se proporcionó imagen.

        Los nombres esperados son Granny-Smith, Orange, Royal-Gala y Satsumas.
        Se supone que las entradas ya fueron validadas; esta función no carga
        modelos ni realiza predicciones.

    Salida esperada al completar el ejercicio:
        Un diccionario con dos claves:
        - "producto": nombre seleccionado, o None si no se puede resolver.
        - "estado": "coincidencia", "solo_texto", "solo_imagen",
          "conflicto" o "sin_producto".

        Si las fuentes discrepan o ambas están ausentes, el producto debe
        ser None. Así, quien llama puede pedir una aclaración.

    """
    if producto_texto == "no especificado":
        producto_texto = None

    if producto_texto is None and producto_imagen is None:
        return {"producto": None, "estado": "sin_producto"}

    if producto_texto is not None and producto_imagen is None:
        return {"producto": producto_texto, "estado": "solo_texto"}

    if producto_texto is None and producto_imagen is not None:
        return {"producto": producto_imagen, "estado": "solo_imagen"}

    if producto_texto == producto_imagen:
        return {"producto": producto_texto, "estado": "coincidencia"}

    if producto_texto != producto_imagen:
        return {"producto": None, "estado": "conflicto"}


def procesar_consulta(mensaje, imagen=None):
    """Combina texto y una imagen PIL/ruta usando los modelos locales.

    Sin texto, la consulta predeterminada es la promoción del producto visual.
    Cada llamada es independiente: no hereda productos de mensajes anteriores.
    """
    if not isinstance(mensaje, str):
        raise ValueError("El mensaje debe ser texto.")
    mensaje = mensaje.strip()
    if not mensaje and imagen is None:
        raise ValueError("Escribe una pregunta o adjunta una imagen.")
    mensaje = mensaje or "¿Qué promoción tiene este producto?"
    producto_texto = nlp.detectar_entidades(mensaje)["producto"]
    producto_imagen = (
        vision.predecir_imagen(imagen)["producto"] if imagen is not None else None
    )
    resultado = resolver_producto(producto_texto, producto_imagen)
    intencion = str(nlp.detectar_intencion_ml(mensaje))
    if resultado["estado"] == "conflicto":
        respuesta = (
            f"Mencionaste {producto_texto}, pero la imagen indica {producto_imagen}. "
            "¿Qué producto quieres consultar? Envía su nombre sin imagen o corrige la foto."
        )
    elif resultado["estado"] == "sin_producto" and intencion not in (
        "saludo", "despedida", "desconocida"
    ):
        respuesta = "¿Qué producto quieres consultar? Escribe su nombre o adjunta una imagen."
    else:
        respuesta = nlp.generar_respuesta(
            mensaje, imagen=resultado["producto"]
        )["respuesta"]
    return {"intencion": intencion, **resultado, "respuesta": respuesta}
