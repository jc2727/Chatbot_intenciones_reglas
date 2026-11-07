"""Integración Flask → agente → NLP/visión; sin entrenamiento ni red."""

import io
import unittest
from unittest.mock import patch

from PIL import Image

import agent
from api import create_app
from catalogo import CLASES


def imagen_png():
    archivo = io.BytesIO()
    Image.new("RGB", (32, 32), "orange").save(archivo, format="PNG")
    archivo.seek(0)
    return archivo, "fruta.png"


class IntegracionTest(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"estado": "ok"})

    def test_solo_texto_modelo_real(self):
        response = self.client.post("/consultar", json={"mensaje": "¿Qué promoción tiene la naranja?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json), {"intencion", "producto", "estado", "respuesta"})
        self.assertEqual(response.json["intencion"], "consultar_promocion")
        self.assertEqual(response.json["producto"], "Orange")
        self.assertEqual(response.json["estado"], "solo_texto")
        self.assertIn("3x2", response.json["respuesta"])

    def test_solo_imagen_modelos_reales(self):
        # Imagen sintética: verifica el recorrido completo, no la exactitud visual.
        response = self.client.post("/consultar", data={"imagen": imagen_png()})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["estado"], "solo_imagen")
        self.assertEqual(response.json["intencion"], "consultar_promocion")
        self.assertIn(response.json["producto"], CLASES)
        self.assertIn(response.json["producto"], response.json["respuesta"])

    def test_texto_e_imagen_coinciden(self):
        with patch("agent.vision.predecir_imagen", return_value={"producto": "Orange"}):
            response = self.client.post("/consultar", data={
                "mensaje": "¿Qué promoción tiene la naranja?", "imagen": imagen_png(),
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["estado"], "coincidencia")
        self.assertEqual(response.json["producto"], "Orange")
        self.assertIn("3x2", response.json["respuesta"])

    def test_imagen_resuelve_producto_ausente(self):
        with patch("agent.vision.predecir_imagen", return_value={"producto": "Satsumas"}):
            response = self.client.post("/consultar", data={
                "mensaje": "¿Qué promoción tiene este producto?", "imagen": imagen_png(),
            })
        self.assertEqual(response.json["estado"], "solo_imagen")
        self.assertIn("15%", response.json["respuesta"])

    def test_conflicto_pide_aclaracion(self):
        with patch("agent.vision.predecir_imagen", return_value={"producto": "Satsumas"}), \
                patch("agent.nlp.generar_respuesta") as generar:
            response = self.client.post("/consultar", data={
                "mensaje": "¿Qué promoción tiene la naranja?", "imagen": imagen_png(),
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["estado"], "conflicto")
        self.assertIsNone(response.json["producto"])
        self.assertIn("Orange", response.json["respuesta"])
        self.assertIn("Satsumas", response.json["respuesta"])
        generar.assert_not_called()

    def test_sin_producto_pide_aclaracion(self):
        response = self.client.post("/consultar", json={"mensaje": "¿Qué promoción tiene este producto?"})
        self.assertEqual(response.json["estado"], "sin_producto")
        self.assertIsNone(response.json["producto"])
        self.assertIn("¿Qué producto", response.json["respuesta"])

    def test_saludo_y_despedida_conservados(self):
        for mensaje, intencion in [("hola", "saludo"), ("adios", "despedida")]:
            with self.subTest(mensaje=mensaje):
                response = self.client.post("/consultar", json={"mensaje": mensaje})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json["intencion"], intencion)
                self.assertTrue(response.json["respuesta"])

    def test_vigencia_y_ejecucion_originales(self):
        for mensaje, fragmento in [
            ("¿Cuándo termina la oferta de Satsumas?", "2026-11-15"),
            ("¿Cómo debe ser la ejecución de naranja?", "cítricos"),
        ]:
            with self.subTest(mensaje=mensaje):
                response = self.client.post("/consultar", json={"mensaje": mensaje})
                self.assertEqual(response.status_code, 200)
                self.assertIn(fragmento, response.json["respuesta"])

    def test_json_invalido(self):
        for datos in [{}, {"mensaje": "  "}, {"mensaje": None}, {"mensaje": 42},
                      {"mensaje": []}, {"mensaje": "x" * 4001}, [],
                      {"mensaje": "hola", "imagen": "/tmp/imagen.png"}]:
            with self.subTest(datos=str(datos)[:70]):
                response = self.client.post("/consultar", json=datos)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.json)

    def test_imagen_invalida(self):
        response = self.client.post("/consultar", data={
            "imagen": (io.BytesIO(b"not an image"), "fruta.png"),
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)

    def test_errores_http_y_limite(self):
        cases = [
            self.client.post("/consultar", data="hola", content_type="text/plain"),
            self.client.post("/consultar", data="{", content_type="application/json"),
            self.client.get("/consultar"),
        ]
        for response, status in zip(cases, [415, 400, 405]):
            self.assertEqual(response.status_code, status)
            self.assertIn("error", response.json)
        self.app.config["MAX_CONTENT_LENGTH"] = 100
        response = self.client.post("/consultar", json={"mensaje": "x" * 101})
        self.assertEqual(response.status_code, 413)
        self.assertIn("error", response.json)

    def test_error_modelo_no_expone_detalles(self):
        with patch("api.procesar_consulta", side_effect=FileNotFoundError("ruta interna")), \
                self.assertLogs(self.app.logger, level="ERROR"):
            response = self.client.post("/consultar", json={"mensaje": "hola"})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("ruta interna", response.json["error"])

    def test_agente_rechaza_consulta_vacia(self):
        for mensaje in [None, 123, " "]:
            with self.subTest(mensaje=mensaje), self.assertRaises(ValueError):
                agent.procesar_consulta(mensaje)


if __name__ == "__main__":
    unittest.main()
