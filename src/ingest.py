"""
ingest.py

Pipeline de ingesta para el RAG sobre el TFM de LAILA.

Pasos:
1. Convertir el PDF a Markdown con pymupdf4llm (conserva una aproximación
   de la jerarquía de encabezados a partir del formato visual del PDF).
2. Dividir el Markdown por encabezados (MarkdownHeaderTextSplitter), de forma
   que cada chunk "sabe" en qué sección del documento vive.
3. Subdividir los chunks que aún sean muy largos con un splitter de caracteres
   estándar, para no enviar al LLM contextos excesivos.
4. Generar embeddings con un modelo local (Ollama) e indexar en Chroma.

Ejecutar:
    python -m src.ingest
"""

import pymupdf4llm
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src import config


def pdf_to_markdown(pdf_path) -> str:
    """Convierte el PDF a una cadena Markdown."""
    print(f"Convirtiendo PDF a Markdown: {pdf_path}")
    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    return md_text


def split_by_headers(md_text: str) -> list[Document]:
    """
    Divide el Markdown por encabezados (#, ##, ###).

    pymupdf4llm tiende a mapear el tamaño de fuente más grande a "#",
    el siguiente a "##", etc. En el TFM esto se corresponde razonablemente
    bien con: # = títulos de sección numerados (1., 2., 3...),
    ## = subsecciones (1.1, 4.3...).
    """
    headers_to_split_on = [
        ("#", "seccion"),
        ("##", "subseccion"),
        ("###", "subsubseccion"),
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  # conservamos el header dentro del contenido del chunk
    )
    docs = splitter.split_text(md_text)
    print(f"Documentos tras split por headers: {len(docs)}")
    return docs


def split_long_chunks(docs: list[Document]) -> list[Document]:
    """
    Subdivide cualquier chunk que supere CHUNK_SIZE, preservando los metadatos
    de sección/subsección que vienen del paso anterior.
    """
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    final_docs = char_splitter.split_documents(docs)
    print(f"Documentos tras subdivisión por tamaño: {len(final_docs)}")
    return final_docs


def build_vectorstore(docs: list[Document]) -> Chroma:
    """Genera embeddings e indexa los documentos en Chroma (persistente en disco)."""
    print(f"Generando embeddings con modelo: {config.EMBEDDING_MODEL}")
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DIR),
    )
    print(f"Vector store persistido en: {config.CHROMA_DIR}")
    return vectorstore


def main():
    md_text = pdf_to_markdown(config.PDF_PATH)
    header_docs = split_by_headers(md_text)
    final_docs = split_long_chunks(header_docs)

    # Vistazo rápido a un par de chunks para verificar que los metadatos
    # de sección se han propagado correctamente.
    print("\n--- Ejemplo de chunk indexado ---")
    print("Metadata:", final_docs[5].metadata)
    print("Contenido (primeros 200 chars):", final_docs[5].page_content[:200])
    print("---------------------------------\n")

    build_vectorstore(final_docs)
    print("Ingesta completada.")


if __name__ == "__main__":
    main()