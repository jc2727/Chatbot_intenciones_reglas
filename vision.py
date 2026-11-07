"""Inferencia ResNet50 en CPU con el checkpoint y transformaciones originales."""

from functools import lru_cache
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50

from catalogo import CLASES

RUTA_MODELO = Path(__file__).resolve().parent / "resnet_frutas.pth-2"

transform_imagen = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@lru_cache(maxsize=1)
def cargar_modelo(ruta=RUTA_MODELO):
    """Reconstruye ResNet50, carga todos los pesos y activa evaluación."""
    modelo = resnet50(weights=None)
    modelo.fc = nn.Linear(modelo.fc.in_features, len(CLASES))
    pesos = torch.load(ruta, map_location="cpu", weights_only=True)
    modelo.load_state_dict(pesos, strict=True)
    modelo.eval()
    return modelo


def predecir_imagen(imagen):
    """Acepta una imagen PIL o ruta; devuelve producto, índice y logits.

    Los logits son puntuaciones sin normalizar, no probabilidades.
    """
    if isinstance(imagen, Image.Image):
        rgb = imagen.convert("RGB")
    else:
        with Image.open(imagen) as archivo:
            rgb = archivo.convert("RGB")
    tensor = transform_imagen(rgb).unsqueeze(0)
    with torch.inference_mode():
        salida = cargar_modelo()(tensor)
    if not torch.isfinite(salida).all().item():
        raise ValueError("El modelo produjo puntuaciones no finitas")
    indice = salida.argmax(dim=1).item()
    return {
        "producto": CLASES[indice],
        "indice": indice,
        "logits": salida[0].tolist(),
    }
