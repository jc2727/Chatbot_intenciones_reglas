# Retail Execution Assistant

Aplicación local: Streamlit → Flask → agente → modelos NLP y ResNet50.
No utiliza LLM, claves externas ni entrenamiento al iniciar.

## Preparación

Usa Python 3.11 y ejecuta desde esta carpeta:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nltk.downloader -d .venv/nltk_data stopwords
```

Conserva los artefactos locales `artifacts/nlp_intenciones.joblib` y
`resnet_frutas.pth-2`. Las versiones de las dependencias de inferencia están
fijadas al entorno existente para mantener compatibilidad con los artefactos.
Los modelos se cargan una vez por proceso, al necesitarlos, y ejecutan en CPU.
No ejecutes `train_nlp.py` para iniciar la aplicación.

## Ejecutar

Terminal 1:

```bash
source .venv/bin/activate
python api.py
```

Terminal 2, en la misma carpeta:

```bash
source .venv/bin/activate
streamlit run app.py --server.address 127.0.0.1
```

Abre http://127.0.0.1:8501. Flask escucha en http://127.0.0.1:5001.
La primera consulta con imagen puede tardar mientras carga ResNet50.
Se usa 5001 porque macOS AirPlay puede ocupar el puerto 5000 y responder 403.
Puedes cambiar el puerto con `RETAIL_API_PORT` en ambas terminales, o cambiar
la dirección completa del backend de Streamlit con `RETAIL_API_URL`.

## Consultas y decisiones

- El agente conserva `resolver_producto()` y las respuestas originales de NLP.
- Texto sin producto: solicita aclaración cuando la intención requiere uno.
- Imagen sin texto: consulta la promoción del producto predicho.
- Texto e imagen diferentes: no selecciona ninguno y solicita aclaración.
- Cada consulta es independiente. El chat conserva el historial visual, pero
  no infiere productos de turnos anteriores. Para resolver un conflicto,
  vuelve a escribir la pregunta con el producto y sin la foto en conflicto.
- El catálogo conserva sus fechas originales; no se comprueba vigencia actual.
  Una predicción visual no incluye umbral de confianza y requiere revisión humana.

`GET /health` comprueba que Flask responde; no verifica la carga de modelos.
`POST /consultar` acepta JSON `{"mensaje": "¿Qué promoción tiene la naranja?"}`
o multipart con los campos `mensaje` e `imagen` (JPEG, PNG o WEBP).
El mensaje admite hasta 4000 caracteres y la solicitud hasta 10 MB.

```bash
curl http://127.0.0.1:5001/health
curl http://127.0.0.1:5001/consultar \
  -H 'Content-Type: application/json' \
  -d '{"mensaje":"¿Qué promoción tiene la naranja?"}'
curl http://127.0.0.1:5001/consultar \
  -F 'mensaje=¿Qué promoción tiene este producto?' -F 'imagen=@fruta.jpg'
```

Respuesta: `intencion`, `producto` (puede ser null), `estado`, `respuesta`.
Los errores incluyen `error` y estado HTTP 400, 413, 415 o 500, según el caso.
Los detalles de fallos internos quedan en el registro de Flask.

## Pruebas

```bash
.venv/bin/python -B -m unittest -v test_componentes test_integracion test_interfaz
```

Las pruebas ejecutan el NLP real y una inferencia visual con imagen sintética
para comprobar los artefactos, sin medir exactitud de clasificación. Para
coincidencias y conflictos se controla la predicción visual, manteniendo el
recorrido HTTP, el NLP y el agente reales. No requieren servicios activos.
La interfaz se prueba con Streamlit AppTest, incluyendo envío al cliente de
Flask, historial y errores de conexión.
