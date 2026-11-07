"""Pruebas de la interfaz sin abrir navegador ni conectar a la red."""

from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import requests
from streamlit.testing.v1 import AppTest

from api import create_app


class InterfazTest(unittest.TestCase):
    def nueva_interfaz(self):
        return AppTest.from_file(str(Path(__file__).with_name("app.py"))).run()

    def test_formulario_consulta_api_y_guarda_conversacion(self):
        client = create_app().test_client()

        def consultar(url, json, timeout):
            self.assertTrue(url.endswith("/consultar"))
            self.assertEqual(timeout, 120)
            response = client.post("/consultar", json=json)
            return Mock(ok=response.status_code == 200, json=lambda: response.json)

        interfaz = self.nueva_interfaz()
        with patch("requests.post", side_effect=consultar):
            interfaz.text_input[0].set_value("¿Qué promoción tiene la naranja?")
            interfaz.button[0].click().run()
        self.assertFalse(interfaz.exception)
        historial = interfaz.session_state["mensajes"]
        self.assertEqual(len(historial), 2)
        self.assertEqual(historial[0]["rol"], "user")
        self.assertIn("3x2", historial[1]["texto"])

    def test_conexion_fallida_muestra_error(self):
        interfaz = self.nueva_interfaz()
        with patch("requests.post", side_effect=requests.ConnectionError):
            interfaz.text_input[0].set_value("hola")
            interfaz.button[0].click().run()
        self.assertFalse(interfaz.exception)
        self.assertIn("No pude conectar", interfaz.error[0].value)
        self.assertEqual(interfaz.session_state["mensajes"], [])

    def test_consulta_vacia_muestra_aviso(self):
        interfaz = self.nueva_interfaz()
        with patch("requests.post") as enviar:
            interfaz.button[0].click().run()
        self.assertFalse(interfaz.exception)
        self.assertTrue(interfaz.warning)
        enviar.assert_not_called()

    def test_airplay_403_sin_json_muestra_diagnostico(self):
        interfaz = self.nueva_interfaz()
        response = Mock(status_code=403, ok=False)
        response.json.side_effect = ValueError("Empty response")
        with patch("requests.post", return_value=response):
            interfaz.text_input[0].set_value("hola")
            interfaz.button[0].click().run()
        self.assertFalse(interfaz.exception)
        self.assertIn("HTTP 403", interfaz.error[0].value)
        self.assertIn("AirPlay", interfaz.error[0].value)
        self.assertEqual(interfaz.session_state["mensajes"], [])

    def test_respuestas_json_inesperadas(self):
        for datos in [[], {}, {"respuesta": None}]:
            with self.subTest(datos=datos):
                interfaz = self.nueva_interfaz()
                with patch("requests.post", return_value=Mock(ok=True, json=lambda: datos)):
                    interfaz.text_input[0].set_value("hola")
                    interfaz.button[0].click().run()
                self.assertFalse(interfaz.exception)
                self.assertTrue(interfaz.error)
                self.assertEqual(interfaz.session_state["mensajes"], [])


if __name__ == "__main__":
    unittest.main()
