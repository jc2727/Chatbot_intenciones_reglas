"""Pruebas pequeñas: .venv/bin/python -B -m unittest -v test_componentes"""

import ast
import json
import math
from pathlib import Path
import unittest

from PIL import Image
import torch

import catalogo
import nlp
import vision


class ComponentesTest(unittest.TestCase):
    def test_datos_conservados_del_notebook(self):
        ruta = Path(__file__).resolve().parent / "Copia_de_Avence_dos+imagen.ipynb"
        notebook = json.loads(ruta.read_text(encoding="utf-8"))
        literales = {}
        for indice in (3, 4, 15):
            tree = ast.parse("".join(notebook["cells"][indice]["source"]))
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in (
                            "datos", "intenciones", "catalogo_productos"
                        ):
                            literales[target.id] = ast.literal_eval(node.value)
        self.assertEqual(catalogo.CATALOGO, literales["datos"])
        self.assertEqual(catalogo.ALIASES, literales["catalogo_productos"])
        self.assertEqual(catalogo.RESPUESTAS, {
            nombre: datos["respuestas"]
            for nombre, datos in literales["intenciones"].items()
        })

    def test_intenciones_con_modelo_guardado(self):
        for mensaje, esperado in [
            ("hola", "saludo"),
            ("¿Qué promoción tiene la naranja?", "consultar_promocion"),
            ("¿Cuándo termina la oferta de Satsumas?", "consultar_vigencia"),
        ]:
            with self.subTest(mensaje=mensaje):
                self.assertEqual(nlp.detectar_intencion_ml(mensaje), esperado)

    def test_entidades_cuatro_frutas_y_ausencia(self):
        for mensaje, esperado in [
            ("manzana verde", "Granny-Smith"),
            ("Royal-Gala", "Royal-Gala"),
            ("NARANJA", "Orange"),
            ("mandarina", "Satsumas"),
            ("hola", "no especificado"),
        ]:
            with self.subTest(mensaje=mensaje):
                self.assertEqual(nlp.detectar_entidades(mensaje)["producto"], esperado)

    def test_respuesta_textual_y_producto_visual(self):
        resultado = nlp.generar_respuesta("¿Qué promoción tiene la naranja?")
        self.assertEqual(resultado["entidades"]["producto"], "Orange")
        self.assertIn("3x2", resultado["respuesta"])
        resultado = nlp.generar_respuesta(
            "¿Qué promoción tiene este producto?", imagen="Satsumas"
        )
        self.assertEqual(resultado["entidades"]["producto"], "Satsumas")
        self.assertIn("15% de descuento", resultado["respuesta"])

    def test_vision_preprocesamiento_y_checkpoint(self):
        # Imagen sintética: verifica integración, no exactitud de clasificación.
        imagen = Image.new("L", (40, 60), color=128)
        tensor = vision.transform_imagen(imagen.convert("RGB"))
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        esperado = torch.tensor([
            (128 / 255 - media) / std
            for media, std in zip(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )
        ])
        torch.testing.assert_close(tensor[:, 0, 0], esperado)
        resultado = vision.predecir_imagen(imagen)
        self.assertEqual(len(resultado["logits"]), 4)
        self.assertTrue(all(math.isfinite(x) for x in resultado["logits"]))
        self.assertEqual(resultado["producto"], catalogo.CLASES[resultado["indice"]])
        self.assertEqual(
            catalogo.CLASES, ("Granny-Smith", "Orange", "Royal-Gala", "Satsumas")
        )
        self.assertFalse(vision.cargar_modelo().training)
        self.assertEqual(next(vision.cargar_modelo().parameters()).device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
