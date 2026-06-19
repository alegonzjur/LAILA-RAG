"""
inspect_store.py

Script de diagnóstico (no forma parte del pipeline de producción).
Permite:
1. Listar TODOS los chunks indexados, para ver cómo quedó el chunking real.
2. Buscar manualmente qué chunk(s) contienen ciertas palabras clave,
   para verificar si la información existe en el store y cómo está fragmentada.
3. Ejecutar una búsqueda por similitud con distintos valores de k,
   para ver qué trae el retriever en cada caso.

Ejecutar:
    python -m src.inspect_store
"""

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from src import config


def load_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )


def list_all_chunks(vectorstore: Chroma):
    """Vuelca todos los chunks con su índice, para inspección manual."""
    data = vectorstore.get()
    docs = data["documents"]
    metadatas = data["metadatas"]
    print(f"Total de chunks en el store: {len(docs)}\n")
    for i, (doc, meta) in enumerate(zip(docs, metadatas)):
        print(f"--- Chunk {i} ---")
        print(f"Metadata: {meta}")
        print(f"Contenido: {doc[:300]}")
        print()


def find_chunks_containing(vectorstore: Chroma, keyword: str):
    """Busca por substring literal (no por similitud semántica) en todos los chunks."""
    data = vectorstore.get()
    docs = data["documents"]
    metadatas = data["metadatas"]
    found = False
    for i, (doc, meta) in enumerate(zip(docs, metadatas)):
        if keyword.lower() in doc.lower():
            found = True
            print(f"--- Encontrado en Chunk {i} ---")
            print(f"Metadata: {meta}")
            print(f"Contenido completo:\n{doc}\n")
    if not found:
        print(f"'{keyword}' no aparece literalmente en ningún chunk indexado.")


def test_similarity_search(vectorstore: Chroma, query: str, k: int = 8):
    """Ejecuta la búsqueda por similitud que haría el retriever, con un k dado."""
    print(f"\nBúsqueda por similitud para: '{query}' (k={k})")
    results = vectorstore.similarity_search_with_score(query, k=k)
    for i, (doc, score) in enumerate(results, start=1):
        preview = doc.page_content[:150].replace("\n", " ")
        print(f"[{i}] score={score:.4f} | {preview}...")


def main():
    vectorstore = load_vectorstore()

    print("=" * 70)
    print("1. Búsqueda literal de 'RTX 3060'")
    print("=" * 70)
    find_chunks_containing(vectorstore, "RTX 3060")

    print("=" * 70)
    print("2. Búsqueda literal de 'GPT-2' + 'medium'")
    print("=" * 70)
    find_chunks_containing(vectorstore, "medium")

    print("=" * 70)
    print("3. Similarity search real para la pregunta de la GPU (k=8)")
    print("=" * 70)
    test_similarity_search(vectorstore, "¿Qué tarjeta gráfica se usó para entrenar los modelos?", k=8)

    print("=" * 70)
    print("4. Similarity search real para la pregunta del modelo NLP (k=8)")
    print("=" * 70)
    test_similarity_search(vectorstore, "¿Qué modelo de generación de texto se utilizó y por qué?", k=8)


if __name__ == "__main__":
    main()