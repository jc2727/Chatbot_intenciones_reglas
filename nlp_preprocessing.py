"""Spanish preprocessing reproduced from Copia_de_Avence_dos+imagen.ipynb."""

import re
import sys
import unicodedata
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

# Keep the downloaded corpus in the active virtual environment.
nltk.data.path.insert(0, str(Path(sys.prefix) / "nltk_data"))

tokenizer = RegexpTokenizer(r"\w+")
stemmer = SnowballStemmer("spanish")
stop_words = set(stopwords.words("spanish"))
stop_words.discard("cuando")


def normalizar_texto(texto):
    """Lowercase, remove accents/punctuation, and collapse whitespace."""
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"[^a-zñ0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def tokenizar_y_stem(texto):
    """Return filtered tokens and their Spanish stems, as in the notebook."""
    texto = normalizar_texto(texto)
    tokens = tokenizer.tokenize(texto)
    tokens = [token for token in tokens if token not in stop_words]
    stems = [stemmer.stem(token) for token in tokens]
    return tokens, stems


def preparar_texto(texto):
    """Produce the stemmed string expected by the fitted BoW pipeline."""
    return " ".join(tokenizar_y_stem(texto)[1])
