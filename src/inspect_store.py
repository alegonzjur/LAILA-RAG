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
    print("Diagnóstico: preguntas exactas generadas por el agente")
    print("(al intentar descomponer 'cuántas canciones rock+pop+rap')")
    print("=" * 70)

    preguntas_agente = [
        "¿Cuántas canciones hay en el conjunto de rock?",
        "¿Cuántas canciones había de pop?",
        "¿Cuántas canciones había de rap?",
    ]

    for pregunta in preguntas_agente:
        test_similarity_search(vectorstore, pregunta, k=10)

    print("\n" + "=" * 70)
    print("Comparación: pregunta más cercana al texto literal del TFM")
    print("=" * 70)
    # El texto real del TFM dice: "Cada dataset tenía el siguiente número
    # de canciones: Rock: 633308 canciones, Pop: 1393559 canciones,
    # Rap: 964605 canciones." Probamos formulaciones más próximas a esa frase.
    test_similarity_search(
        vectorstore,
        "número de canciones por género rock pop rap dataset",
        k=10,
    )


if __name__ == "__main__":
    main()