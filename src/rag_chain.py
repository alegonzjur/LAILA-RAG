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

Modo HyDE (activable desde config.USE_HYDE = True):
    En vez de hacer embedding de la pregunta directamente, se usa el LLM
    para generar primero una "respuesta hipotética" de cómo podría estar
    expresada esa información en el documento, y luego se hace embedding
    de esa respuesta hipotética para buscar en el vector store.

    pregunta → LLM (genera respuesta hipotética) → embedding → Chroma

    Esto mejora el recall cuando hay mismatch de vocabulario entre la
    pregunta (informal, interrogativa) y el documento (técnico, declarativo).
    Ver IMPROVEMENTS.md para el diagnóstico que motivó esta implementación.

Ejecutar como demo manual:
    python -m src.rag_chain
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
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

HYDE_PROMPT_TEMPLATE = """Genera un fragmento breve (2-4 frases) que podría \
aparecer en un Trabajo de Fin de Máster sobre inteligencia artificial y \
generación musical, respondiendo a la siguiente pregunta. No necesita ser \
exacto: solo debe estar escrito en el estilo técnico y el vocabulario que \
usaría ese tipo de documento.

Pregunta: {question}

Fragmento hipotético:"""


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


def build_hyde_retriever(vectorstore: Chroma, llm: ChatOllama):
    """
    Construye un retriever HyDE (Hypothetical Document Embeddings).

    En vez de buscar por similitud con el embedding de la pregunta original,
    este retriever:
    1. Usa el LLM para generar una "respuesta hipotética" a la pregunta,
       expresada en el vocabulario y estilo del documento.
    2. Hace embedding de esa respuesta hipotética.
    3. Busca en Chroma los chunks más similares a ese embedding.

    El resultado es un retriever drop-in (misma interfaz que el estándar)
    que puede usarse exactamente igual en la chain principal.
    """
    hyde_prompt = ChatPromptTemplate.from_template(HYDE_PROMPT_TEMPLATE)

    # Chain que convierte una pregunta en una respuesta hipotética (texto)
    hyde_chain = hyde_prompt | llm | StrOutputParser()

    # Convertimos el vectorstore en un retriever estándar
    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": config.RETRIEVER_K}
    )

    # Retriever HyDE: primero genera la respuesta hipotética,
    # luego la usa como query para el retriever base.
    # RunnableLambda nos permite envolver cualquier función como un Runnable LCEL.
    def retrieve_with_hyde(pregunta: str) -> list[Document]:
        respuesta_hipotetica = hyde_chain.invoke({"question": pregunta})
        print(f"\n[HyDE] Respuesta hipotética generada:\n{respuesta_hipotetica}\n")
        return base_retriever.invoke(respuesta_hipotetica)

    return RunnableLambda(retrieve_with_hyde)


def build_rag_chain():
    """
    Construye y devuelve la chain completa de RAG.

    Si config.USE_HYDE es True, usa el retriever HyDE.
    Si es False, usa el retriever estándar por similitud directa.

    Devuelve un Runnable que, al invocarse con un string (la pregunta),
    produce un dict: {"answer": str, "context": list[Document]}
    """
    vectorstore = load_vectorstore()
    llm = ChatOllama(model=config.LLM_MODEL, temperature=0.1)

    if config.USE_HYDE:
        print("[RAG] Modo HyDE activado.")
        retriever = build_hyde_retriever(vectorstore, llm)
    else:
        print("[RAG] Modo estándar (sin HyDE).")
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": config.RETRIEVER_K}
        )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # Sub-chain de generación
    generation_chain = prompt | llm | StrOutputParser()

    # Chain completa con RunnableParallel para devolver respuesta + fuentes
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
        seccion = doc.metadata.get("subseccion") or doc.metadata.get(
            "seccion", "sin sección"
        )
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