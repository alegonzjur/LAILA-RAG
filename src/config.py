"""
Configuración centralizada del proyecto.
Mantener aquí todos los parámetros facilita experimentar
(p. ej. cambiar de modelo o de chunk_size) sin tocar el resto del código.
"""

from pathlib import Path

# --- Rutas ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "TFM_LAILA.pdf"
CHROMA_DIR = BASE_DIR / "chroma_db"

# --- Modelos Ollama ---
LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

# --- Chunking ---
# Tamaño máximo de un chunk "hijo" tras el split por headers.
# Algunas secciones del TFM (p.ej. la lista de hiperparámetros) son largas
# y conviene subdividirlas, pero conservando el contexto de a qué sección pertenecen.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Nombre de la colección en Chroma
COLLECTION_NAME = "laila_tfm"

# --- Retrieval ---
RETRIEVER_K = 4  # nº de chunks a recuperar por consulta 
