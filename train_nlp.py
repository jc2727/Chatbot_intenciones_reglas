"""Train only the notebook's BoW + LinearSVC model and verify persistence.

Run: .venv/bin/python -B train_nlp.py
The saved pipeline expects text processed by nlp_preprocessing.preparar_texto.
"""

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

import joblib
import nltk
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from nlp_preprocessing import preparar_texto, stop_words

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Copia_de_Avence_dos+imagen.ipynb"
ARTIFACT = ROOT / "artifacts" / "nlp_intenciones.joblib"


def leer_intenciones():
    """Read the literal dictionary without executing any notebook cells."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "intenciones"
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise ValueError("No literal intenciones dictionary found in the notebook")


def main():
    intenciones = leer_intenciones()
    # Preserve dictionary order and example order, matching the original rows.
    textos = []
    etiquetas = []
    for nombre, datos in intenciones.items():
        for ejemplo in datos["ejemplos"]:
            textos.append(ejemplo)
            etiquetas.append(nombre)
    if len(textos) != 100 or sorted(Counter(etiquetas).values()) != [20] * 5:
        raise ValueError("Expected the original 100 examples, 20 per intention")

    procesados = [preparar_texto(texto) for texto in textos]
    X_train, X_test, y_train, y_test = train_test_split(
        procesados, etiquetas, test_size=0.2, random_state=42, stratify=etiquetas
    )
    pipeline = Pipeline([
        ("vectorizador", CountVectorizer(ngram_range=(1, 2))),
        ("modelo", LinearSVC()),
    ])
    pipeline.fit(X_train, y_train)
    predicciones = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predicciones),
        "precision_weighted": precision_score(y_test, predicciones, average="weighted"),
        "recall_weighted": recall_score(y_test, predicciones, average="weighted"),
        "f1_weighted": f1_score(y_test, predicciones, average="weighted"),
    }
    # Metadata does not change the fitted estimator or its hyperparameters.
    pipeline.training_metadata_ = {
        "source_notebook": NOTEBOOK.name,
        "source_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "nltk": nltk.__version__,
            "joblib": joblib.__version__,
            "numpy": np.__version__,
        },
        "preprocessing": "nlp_preprocessing.preparar_texto (outside pipeline)",
        "stop_words": sorted(stop_words),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "split_random_state": 42,
        "metrics": metrics,
    }
    ARTIFACT.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, ARTIFACT)
    restored = joblib.load(ARTIFACT)
    # Include every original example, not just the held-out split.
    np.testing.assert_array_equal(
        pipeline.predict(procesados), restored.predict(procesados)
    )
    np.testing.assert_array_equal(predicciones, restored.predict(X_test))

    print("NEW RUN (not historical notebook results)")
    print(json.dumps(pipeline.training_metadata_["versions"], indent=2))
    print(f"Training examples: {len(X_train)}; test examples: {len(X_test)}")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, predicciones, zero_division=0))
    print("Confusion matrix class order:", pipeline.classes_.tolist())
    print(confusion_matrix(y_test, predicciones, labels=pipeline.classes_))
    print("Persistence check PASSED: identical predictions on all 100 examples and test split")
    print("Saved:", ARTIFACT)


if __name__ == "__main__":
    main()
