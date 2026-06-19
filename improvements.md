# Mejoras pendientes / Limitaciones conocidas

Este documento recoge decisiones tomadas durante el desarrollo que quedan
abiertas para iterar más adelante, junto con el razonamiento técnico de por
qué ocurren.

## 1. Metadatos de sección incorrectos tras la conversión PDF → Markdown

**Estado:** Pendiente de arreglar. Documentado, no bloqueante.

**Qué pasa:**
`pymupdf4llm` infiere el nivel de encabezado Markdown (`#`, `##`, `###`) a
partir del **tamaño de fuente visual** del texto en el PDF, no de una
jerarquía semántica real. El título de portada del TFM
("Generación Automática de Canciones...") está en una fuente más grande que
cualquier encabezado de sección del cuerpo del documento, por lo que se
infiere como el único header de nivel 1 (`#`) de todo el documento.

Consecuencia: todas las secciones numeradas reales (1. Introducción,
1.1 Problema a resolver, 2. Objetivos...) caen como nivel 2 (`##`) o más
profundo, heredando el título de portada como su "sección padre" en el
metadato `seccion`, en lugar de heredar la sección numerada correspondiente
(ej. "1. Introducción").

El contenido textual de cada chunk es correcto; el problema es únicamente
de metadatos.

**Por qué no es bloqueante ahora mismo:**
El RAG no depende de estos metadatos para recuperar o generar respuestas
(el retrieval funciona sobre el embedding del contenido, no sobre la
metadata de sección). El impacto es solo en la posibilidad de filtrar por
sección o de citar "de qué apartado viene esta respuesta" con precisión.

**Posibles soluciones a evaluar:**
- Eliminar/saltar la página de portada antes de convertir a Markdown
  (ej. extraer solo a partir de la página 2 con PyMuPDF antes de pasar a
  `pymupdf4llm`).
- Post-procesar el Markdown generado con una regex que detecte el patrón
  real de las secciones del TFM (`^\d+\.\s`, `^\d+\.\d+\s`) y re-mapee los
  niveles de header en función de ese patrón, ignorando el tamaño de fuente.
- Cambiar de estrategia: usar `PyPDFLoader` + chunking manual basado en
  regex sobre el texto plano, en vez de pasar por Markdown (opción 2 que se
  descartó inicialmente en favor de la conversión a Markdown).

# Mejoras pendientes / Limitaciones conocidas

Este documento recoge decisiones tomadas durante el desarrollo que quedan
abiertas para iterar más adelante, junto con el razonamiento técnico de por
qué ocurren.

## 1. Metadatos de sección incorrectos tras la conversión PDF → Markdown

**Estado:** Pendiente de arreglar. Documentado, no bloqueante.

**Qué pasa:**
`pymupdf4llm` infiere el nivel de encabezado Markdown (`#`, `##`, `###`) a
partir del **tamaño de fuente visual** del texto en el PDF, no de una
jerarquía semántica real. El título de portada del TFM
("Generación Automática de Canciones...") está en una fuente más grande que
cualquier encabezado de sección del cuerpo del documento, por lo que se
infiere como el único header de nivel 1 (`#`) de todo el documento.

Consecuencia: todas las secciones numeradas reales (1. Introducción,
1.1 Problema a resolver, 2. Objetivos...) caen como nivel 2 (`##`) o más
profundo, heredando el título de portada como su "sección padre" en el
metadato `seccion`, en lugar de heredar la sección numerada correspondiente
(ej. "1. Introducción").

El contenido textual de cada chunk es correcto; el problema es únicamente
de metadatos.

**Por qué no es bloqueante ahora mismo:**
El RAG no depende de estos metadatos para recuperar o generar respuestas
(el retrieval funciona sobre el embedding del contenido, no sobre la
metadata de sección). El impacto es solo en la posibilidad de filtrar por
sección o de citar "de qué apartado viene esta respuesta" con precisión.

**Posibles soluciones a evaluar:**
- Eliminar/saltar la página de portada antes de convertir a Markdown
  (ej. extraer solo a partir de la página 2 con PyMuPDF antes de pasar a
  `pymupdf4llm`).
- Post-procesar el Markdown generado con una regex que detecte el patrón
  real de las secciones del TFM (`^\d+\.\s`, `^\d+\.\d+\s`) y re-mapee los
  niveles de header en función de ese patrón, ignorando el tamaño de fuente.
- Cambiar de estrategia: usar `PyPDFLoader` + chunking manual basado en
  regex sobre el texto plano, en vez de pasar por Markdown (opción 2 que se
  descartó inicialmente en favor de la conversión a Markdown).

## 2. Trade-off recall vs. precisión al ajustar RETRIEVER_K

**Estado:** Documentado como limitación conocida del enfoque actual. No bloqueante.

**Diagnóstico realizado:**
Con `RETRIEVER_K = 4`, dos preguntas de prueba fallaban por bajo recall:
el retriever no traía el chunk que contenía la respuesta literal.
Usando `src/inspect_store.py` para hacer `similarity_search_with_score`
directamente sobre Chroma (sin pasar por el LLM), se confirmó que:

- El chunk con "NVIDIA GeForce RTX 3060" aparecía en la **posición 8** del
  ranking por similitud (score 0.6414, muy cercano al de la posición 1,
  0.6043 — es decir, compitiendo en un rango muy estrecho con chunks que
  solo hablan *alrededor* del tema sin contener el dato).
- El chunk con "se utilizará la versión 'medium' [de GPT-2]" **no aparecía
  ni en el top-8**. Este chunk vive dentro de una subsección larga y
  argumentativa (comparación de Llama vs GPT-2), y su embedding se parece
  más a "fragmento de cierre de una comparación" que a la formulación
  literal de la pregunta.

**Cambio aplicado:** `RETRIEVER_K` subido de 4 a 8.

**Resultado tras el cambio (re-test con la chain completa, no solo el
similarity search crudo):**
- ✅ Pregunta sobre la GPU: se resolvió correctamente.
- ❌ Pregunta sobre el modelo NLP (GPT-2 medium): sigue sin resolverse,
  porque el chunk relevante no estaba ni en el top-8 del ranking crudo;
  subir k a 8 no era suficiente para este caso.
- ⚠️ Efecto colateral nuevo en una pregunta que SÍ funcionaba con k=4:
  "¿Cuántas canciones tenía el dataset de pop tras el filtrado por
  idioma?" pasó de responder correctamente (1.393.559, el total tras
  filtrar por idioma) a responder 1.254.203 (el subconjunto de
  *entrenamiento* tras el split 90/10, un número distinto que también
  aparece literalmente en el documento). Con k=4 el contexto no contenía
  ese segundo número y no había ambigüedad; con k=8 el LLM tuvo varios
  números de canciones distintos en el contexto y escogió el incorrecto
  para la pregunta formulada.

**Conclusión:** subir k mejora el recall (encuentra más chunks
potencialmente relevantes) pero no es gratis: más contexto introduce más
ruido y aumenta el riesgo de que el LLM mezcle datos numéricos similares
que aparecen en distintos chunks del mismo contexto. Es un trade-off
real de RAG, no un bug a "arreglar" con un valor mágico de k.

**Posibles mejoras a futuro (no aplicadas aún):**
- Aumentar `CHUNK_OVERLAP` para que la frase de decisión final dentro de
  secciones argumentativas largas (caso GPT-2 medium) aparezca repetida
  en más de un chunk, aumentando su probabilidad de ser recuperada.
- Re-ranking: recuperar con k alto (ej. 10-12) y aplicar un segundo paso
  de re-ranking (ej. con un cross-encoder) para quedarnos solo con los
  3-4 chunks realmente más relevantes antes de pasarlos al LLM — así se
  obtiene el recall de k alto sin el ruido de meter todo al prompt.
- Prompt engineering: instruir explícitamente al LLM a distinguir entre
  cifras que respondan exactamente a la pregunta formulada (p. ej.
  "total" vs. "subconjunto de entrenamiento") cuando el contexto
  contenga varios números similares.

## 3. (Espacio para próximas entradas)