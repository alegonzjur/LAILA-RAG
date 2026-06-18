"""
rag_chain.py

Construye la chain de RAG sobre el TFM de LAILA usando LCEL.

Composición de la chain:

    {"context": retriever | format_docs, "question": passthrough}
        | prompt
        | llm
        | parser

Pero como además queremos devolver las FUENTES (los chunks usados, con su
metadata de sección y página) junto a la respuesta, envolvemos la chain de
generación dentro de un RunnableParallel que ejecuta dos ramas:

    - "answer": la chain de generación de texto de toda la vida
    - "context": los documentos recuperados, sin tocar, para mostrarlos como fuente

Esto es un patrón típico en RAG real: separar "lo que se le manda al LLM
como texto" de "lo que se le muestra al usuario como trazabilidad".

Ejecutar como demo manual:
    python -m src.rag_chain
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

from src import config


PROMPT_TEMPLATE = """Eres un asistente que responde preguntas únicamente \
basándose en el contexto proporcionado, extraído del Trabajo de Fin de \
Máster (TFM) titulado "Generación Automática de Canciones mediante Modelos \
de Lenguaje Natural y Redes Neuronales" (proyecto LAILA).

Reglas:
- Responde solo con información que aparezca en el contexto.
- Si el contexto no contiene la respuesta, dilo explícitamente: \
no inventes información ni la completes con conocimiento general.
- Responde en español, de forma clara y concisa.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""


def format_docs(docs: list[Document]) -> str:
    """Concatena el contenido de los documentos recuperados en un solo bloque de texto."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def load_vectorstore() -> Chroma:
    """Carga el vector store de Chroma ya persistido (no vuelve a indexar nada)."""
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )
    return vectorstore


def build_rag_chain():
    """
    Construye y devuelve la chain completa de RAG.

    Devuelve un Runnable que, al invocarse con un string (la pregunta),
    produce un dict: {"answer": str, "context": list[Document]}
    """
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_K})

    llm = ChatOllama(model=config.LLM_MODEL, temperature=0.1)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # Sub-chain de generación: toma {"context": str, "question": str} -> respuesta en texto
    generation_chain = prompt | llm | StrOutputParser()

    # Chain completa: recupera documentos UNA vez y los reutiliza
    # tanto para generar la respuesta como para devolverlos como fuente.
    rag_chain = RunnableParallel(
        {
            "context_docs": retriever,
            "question": RunnablePassthrough(),
        }
    ) | RunnableParallel(
        {
            "answer": (
                {
                    "context": lambda x: format_docs(x["context_docs"]),
                    "question": lambda x: x["question"],
                }
                | generation_chain
            ),
            "context": lambda x: x["context_docs"],
        }
    )

    return rag_chain


def format_sources(docs: list[Document]) -> str:
    """Formatea los documentos fuente de forma legible para mostrar al usuario."""
    lines = []
    for i, doc in enumerate(docs, start=1):
        seccion = doc.metadata.get("subseccion") or doc.metadata.get("seccion", "sin sección")
        pagina = doc.metadata.get("page", "?")
        preview = doc.page_content[:120].replace("\n", " ")
        lines.append(f"[{i}] ({seccion}, pág. {pagina}) {preview}...")
    return "\n".join(lines)


def main():
    chain = build_rag_chain()

    preguntas_demo = [
        "¿Qué modelo de generación de texto se utilizó y por qué?",
        "¿Cuántas canciones tenía el dataset de pop tras el filtrado por idioma?",
        "¿Qué tarjeta gráfica se usó para entrenar los modelos?",
    ]

    for pregunta in preguntas_demo:
        print(f"\n{'='*70}\nPREGUNTA: {pregunta}\n{'='*70}")
        result = chain.invoke(pregunta)
        print(f"\nRESPUESTA:\n{result['answer']}")
        print(f"\nFUENTES:\n{format_sources(result['context'])}")


if __name__ == "__main__":
    main()