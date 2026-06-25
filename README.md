# laila-rag

Sistema RAG (Retrieval-Augmented Generation) completamente local, construido
sobre el TFM *"Generación Automática de Canciones mediante Modelos de Lenguaje
Natural y Redes Neuronales"* (proyecto LAILA) como documento de conocimiento.

Permite hacer preguntas en lenguaje natural sobre el contenido del TFM y
recibir respuestas fundamentadas en el texto real del documento, junto con
las fuentes concretas usadas para generar cada respuesta.

Desarrollado como proyecto de portfolio para demostrar comprensión práctica
de los componentes de un sistema RAG y los trade-offs reales que aparecen
al construirlo: chunking, retrieval, generación, agentes con tools y técnicas
avanzadas de retrieval como HyDE.

> **Todo el sistema corre en local, sin API keys de pago.**
> Modelos servidos por [Ollama](https://ollama.com).

---

## Stack

| Componente | Tecnología |
|---|---|
| LLM (RAG chain) | `llama3.2:3b` vía Ollama |
| LLM (agente) | `llama3.1:8b` vía Ollama |
| Embeddings | `nomic-embed-text` vía Ollama |
| Vector store | ChromaDB (persistente en disco) |
| Orquestación | LangChain 1.x + LCEL |
| Conversión PDF | `pymupdf4llm` |
| Splitting | `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` |

---

## Estructura del proyecto

```
laila-rag/
├── data/
│   └── TFM_LAILA.pdf           # Documento fuente
├── src/
│   ├── config.py               # Parámetros centralizados (modelos, rutas, chunking, flags)
│   ├── ingest.py               # Pipeline de ingesta: PDF → Markdown → chunks → Chroma
│   ├── rag_chain.py            # RAG chain con LCEL (respuesta + fuentes, modo HyDE opcional)
│   ├── agent.py                # Agente con dos tools: RAG + calculadora segura
│   └── inspect_store.py        # Herramienta de diagnóstico del vector store
├── chroma_db/                  # Vector store persistido (generado por ingest.py, no versionado)
├── IMPROVEMENTS.md             # Limitaciones conocidas, diagnósticos y decisiones documentadas
├── requirements.txt
└── README.md
```

---

## Instalación y uso

### Requisitos previos

- Python 3.10+
- [Ollama](https://ollama.com/download) instalado y corriendo

### 1. Clonar el repositorio

```bash
git clone https://github.com/alegonzjur/laila-rag.git
cd laila-rag
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
# Windows:
.\venv\Scripts\Activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Descargar los modelos de Ollama

```bash
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 4. Indexar el documento

```bash
python -m src.ingest
```

Esto convierte el PDF a Markdown, lo divide en chunks respetando la
estructura de encabezados, genera embeddings y persiste el vector store
en `chroma_db/`. Solo es necesario ejecutarlo una vez.

### 5. Usar la RAG chain

```bash
python -m src.rag_chain
```

Ejecuta tres preguntas de demo sobre el TFM y muestra la respuesta junto
con las fuentes recuperadas. Para cambiar las preguntas, edita el array
`preguntas_demo` en `src/rag_chain.py`.

Para activar el modo **HyDE** (Hypothetical Document Embeddings), cambia
en `src/config.py`:

```python
USE_HYDE = True
```

### 6. Usar el agente

```bash
python -m src.agent
```

El agente puede responder preguntas sobre el TFM (tool `buscar_en_tfm`) y
realizar cálculos sobre los datos recuperados (tool `calculadora`). El
rastro completo de razonamiento y tool calls se imprime en consola.

---

## Decisiones de diseño relevantes

### Chunking: PDF → Markdown → split por headers

En vez de chunking por número fijo de caracteres, el pipeline convierte el
PDF a Markdown con `pymupdf4llm` (que infiere headers a partir del tamaño
de fuente) y luego aplica `MarkdownHeaderTextSplitter` para dividir
respetando la jerarquía de secciones del documento. Los chunks resultantes
llevan metadatos de sección (`seccion`, `subseccion`) que permiten trazar
de qué parte del TFM viene cada respuesta.

**Limitación conocida:** `pymupdf4llm` infiere los headers a partir del
tamaño visual de fuente, no de una jerarquía semántica real. El título de
portada (fuente más grande que cualquier sección) se detecta como el único
header de nivel 1, haciendo que todas las secciones reales hereden ese
título como "sección padre". El contenido es correcto; los metadatos de
sección, parcialmente imprecisos.

### Dos modelos distintos: RAG vs. agente

El pipeline RAG usa `llama3.2:3b` (ligero, suficiente para una sola pasada
de generación con contexto ya recuperado). El agente usa `llama3.1:8b`
porque el tool-calling multi-paso (decidir qué tool llamar, cuándo, y cómo
combinar los resultados) exige más capacidad de razonamiento. Con `llama3.2:3b`
el agente generaba tool calls mal formadas y perdía el hilo tras la primera
llamada.

### HyDE como mejora del retrieval

Se implementó HyDE (`src/rag_chain.py`, `build_hyde_retriever()`) para
atacar el mismatch de vocabulario entre preguntas en lenguaje informal y
el texto técnico del TFM. Los resultados fueron mixtos: HyDE mejoró el
recall para preguntas sobre cifras del dataset, pero perjudicó en otros
casos donde el LLM generó una hipótesis con vocabulario incorrecto
(ej. "CNN para música" en vez de GPT-2), desviando al retriever. El flag
`USE_HYDE` permite comparar ambos modos directamente.

### Calculadora sin `eval()`

La tool `calculadora` del agente usa el módulo `ast` para parsear y evaluar
expresiones aritméticas nodo a nodo, con una lista blanca de operadores
permitidos. Esto evita que una expresión generada por el LLM (o una
inyección de prompt) pudiera ejecutar código arbitrario vía `eval()`.

---

## Hallazgos y limitaciones documentadas

El archivo [`IMPROVEMENTS.md`](./IMPROVEMENTS.md) recoge en detalle cuatro
limitaciones identificadas durante el desarrollo, junto con el proceso de
diagnóstico, los experimentos realizados para verificar las hipótesis y las
posibles mejoras a futuro:

1. **Metadatos de sección imprecisos** por la conversión PDF→Markdown.
2. **Trade-off recall/precisión** al ajustar `RETRIEVER_K`: subir k mejora
   el recall pero introduce ruido que puede confundir al LLM con datos
   numéricos similares.
3. **Few-shot leakage** desde el docstring de una tool: el LLM usó las
   cifras del ejemplo de documentación de `calculadora` como si fueran
   datos reales recuperados, dando un resultado "correcto" por una razón
   completamente equivocada.
4. **HyDE: resultados mixtos** — útil para mismatch de vocabulario, pero
   contraproducente cuando el LLM no tiene suficiente conocimiento del
   dominio para generar una hipótesis plausible.

---

## Autor

**Alejandro González Jurado**
[GitHub](https://github.com/alegonzjur) ·
[Portfolio](https://alegonzjur.github.io)