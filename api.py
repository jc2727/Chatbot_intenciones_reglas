"""API local: python api.py."""

import os
import warnings

from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

from agent import procesar_consulta


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

    @app.get("/health")
    def health():
        # Liveness only; model loading is cached and deferred until first use.
        return jsonify(estado="ok")

    @app.post("/consultar")
    def consultar():
        imagen = None
        if request.is_json:
            datos = request.get_json()
            if not isinstance(datos, dict):
                return jsonify(error="El cuerpo JSON debe ser un objeto."), 400
            if "imagen" in datos:
                return jsonify(error="Envía imágenes mediante multipart/form-data."), 400
            mensaje = datos.get("mensaje", "")
        elif request.mimetype == "multipart/form-data":
            mensaje = request.form.get("mensaje", "")
            archivo = request.files.get("imagen")
            if archivo is not None:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", Image.DecompressionBombWarning)
                        with Image.open(archivo.stream) as original:
                            if original.format not in ("JPEG", "PNG", "WEBP"):
                                return jsonify(error="Usa una imagen JPEG, PNG o WEBP."), 400
                            imagen = original.convert("RGB")
                except (UnidentifiedImageError, OSError, ValueError,
                        Image.DecompressionBombError, Image.DecompressionBombWarning):
                    return jsonify(error="La imagen no es válida o es demasiado grande."), 400
        else:
            return jsonify(error="Usa JSON o multipart/form-data."), 415

        if not isinstance(mensaje, str) or len(mensaje) > 4000:
            return jsonify(error="El mensaje debe ser texto de hasta 4000 caracteres."), 400
        if not mensaje.strip() and imagen is None:
            return jsonify(error="Escribe una pregunta o adjunta una imagen."), 400
        try:
            return jsonify(procesar_consulta(mensaje, imagen))
        finally:
            if imagen is not None:
                imagen.close()

    @app.errorhandler(HTTPException)
    def http_error(error):
        mensaje = "La solicitud supera el límite de 10 MB." if error.code == 413 else error.description
        return jsonify(error=mensaje), error.code

    @app.errorhandler(Exception)
    def unexpected_error(error):
        app.logger.exception("No se pudo procesar la consulta")
        return jsonify(error="No se pudo procesar la consulta. Revisa los modelos y el registro del servidor."), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("RETAIL_API_PORT", "5001")), debug=False)
