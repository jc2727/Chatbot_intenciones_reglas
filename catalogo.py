"""Catálogo y respuestas del notebook Copia_de_Avence_dos+imagen.

Las fechas y promociones son los datos originales; no implican vigencia actual.
"""

CLASES = ("Granny-Smith", "Orange", "Royal-Gala", "Satsumas")

CATALOGO = [
    {
        "producto": "Granny-Smith",
        "promocion": "20% de descuento",
        "fecha_inicio": "2026-10-01",
        "fecha_fin": "2026-10-31",
        "ejecucion": "Exhibición en sección de frutas con precio promocional visible",
    },
    {
        "producto": "Royal-Gala",
        "promocion": "2x1",
        "fecha_inicio": "2026-10-05",
        "fecha_fin": "2026-10-25",
        "ejecucion": "Exhibición en sección de manzanas con señalización promocional",
    },
    {
        "producto": "Orange",
        "promocion": "3x2",
        "fecha_inicio": "2026-10-10",
        "fecha_fin": "2026-11-10",
        "ejecucion": "Exhibición en sección de cítricos con material promocional",
    },
    {
        "producto": "Satsumas",
        "promocion": "15% de descuento",
        "fecha_inicio": "2026-10-15",
        "fecha_fin": "2026-11-15",
        "ejecucion": "Exhibición en sección de mandarinas con precio promocional visible",
    },
]

ALIASES = {
    "Granny-Smith": ["granny smith", "granny-smith", "manzana verde"],
    "Royal-Gala": ["royal gala", "royal-gala", "manzana gala"],
    "Orange": ["orange", "naranja", "naranjas"],
    "Satsumas": ["satsumas", "satsuma", "mandarina", "mandarinas"],
}

RESPUESTAS = {
    "saludo": [
        "hola cómo te ayudo? dime un producto y qué necesitas saber",
        "buenos dias, soy tu asistente de ejecución dime de qué producto buscas información",
        "Hola soy tu asistente de ejecución en que puedo ayudarte?",
    ],
    "despedida": [
        "Gracias por ayudarnos a tener una gran ejecución",
        "Con gusto, siempre a la orden",
    ],
    "consultar_promocion": [
        "la promoción del producto {producto} es {promocion}",
        "la promoción activa de {producto} es {promocion}",
    ],
    "consultar_ejecucion": [
        "la ejecución del producto {producto} es {ejecucion}",
        "La correcta ejecución adicional del producto {producto} es {ejecucion}",
    ],
    "consultar_vigencia": [
        "la vigencia para el producto {producto} es {vigencia}",
        "la vigencia del descuento del producto {producto} es {vigencia}",
    ],
}

COLUMNAS_INTENCIONES = {
    "consultar_promocion": ["promocion"],
    "consultar_ejecucion": ["ejecucion"],
    "consultar_vigencia": ["fecha_inicio", "fecha_fin"],
}


def buscar_producto(producto):
    """Devuelve una copia del registro, o None si no existe."""
    for fila in CATALOGO:
        if fila["producto"] == producto:
            return fila.copy()
    return None
