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

## 3. Few-shot leakage: el ejemplo del docstring de una tool fue usado como dato real

**Estado:** Causa raíz identificada y corregida. Útil registrar el caso
porque ilustra un fallo de diseño de prompt, no del modelo ni del retriever.

**Síntoma observado:**
Al pedirle al agente sumar las cifras de canciones de rock, pop y rap
(`agent.py`, con `AGENT_MODEL = "llama3.1:8b"`), el agente devolvió el
resultado numérico correcto (2.991.472) **a pesar de que las tres llamadas
a `buscar_en_tfm` habían fallado** (el retriever no encontró las cifras
para esa formulación concreta de la pregunta). El `AIMessage` con las tool
calls mostraba las 4 llamadas (3 búsquedas + 1 cálculo) emitidas en un
único turno, con la calculadora ya conteniendo
`"633308 + 1393559 + 964605"` — las cifras EXACTAS y correctas del TFM —
antes de que existiera ningún resultado de búsqueda en la conversación.

El comportamiento se repitió de forma idéntica en una segunda ejecución
(mismas cifras, mismo orden), lo que descartó que fuera aleatoriedad del
modelo.

**Causa raíz:**
El docstring de la tool `calculadora` (que LangChain expone al LLM como
parte de la descripción de la herramienta, visible en cada turno) incluía
un ejemplo de uso con las cifras REALES de rock/pop/rap del TFM:

    "El argumento debe ser una expresión matemática válida,
     ej: '633308 + 1393559 + 964605'"

El LLM tenía esas cifras delante en el prompt del sistema (como parte del
schema de tools) y las usó directamente como si fueran el resultado de una
búsqueda real, sin esperar ni comprobar las `Observation` de
`buscar_en_tfm`. El acierto del resultado final fue, por tanto, una
consecuencia directa de un error de diseño del prompt (un ejemplo
"de mentira" que coincidía con datos reales), no una capacidad real del
agente para razonar y combinar herramientas correctamente.

Investigación adicional: se confirmó (vía búsqueda web, documentación de
NVIDIA NIM y un PR de LangChain sobre `langchain-oci`) que **Llama 3.1
(8B/70B) no tiene soporte nativo de tool-calling paralelo** — esa
capacidad solo se añadió formalmente a partir de Llama 4. Esto sugiere
que la propia emisión de 4 tool calls en un único turno ya era, de por sí,
un comportamiento fuera de lo que el modelo fue entrenado a hacer de forma
fiable, independientemente del problema del docstring.

**Por qué es un hallazgo importante para discutir en entrevista:**
Demuestra un riesgo real y poco intuitivo en sistemas con tools: el texto
descriptivo de una herramienta (pensado solo como documentación para
desarrolladores) es, en la práctica, **parte del prompt que ve el LLM en
cada turno**. Cualquier dato "de ejemplo" en esa descripción puede ser
tratado por el modelo como información de dominio válida, especialmente
si coincide por casualidad (o por descuido del desarrollador, como en este
caso) con datos reales del contexto de la aplicación. Es una forma sutil
de fuga de información ("few-shot leakage") que no aparece al revisar solo
la respuesta final del agente — solo se detecta inspeccionando el rastro
completo de tool calls y comparándolo con lo que las tools realmente
devolvieron.

**Arreglo aplicado:**
1. Cambiado el ejemplo del docstring de `calculadora` a cifras genéricas
   sin relación con el dominio (`"12 + 34 + 56"`), y añadida una
   instrucción explícita en el propio docstring: "no utilices nunca cifras
   de ejemplo ni cifras que no hayas visto en una Observation".
2. Reforzado `SYSTEM_PROMPT` en `agent.py` con reglas explícitas:
   llamar a las tools de una en una (no en paralelo), esperar el resultado
   de cada búsqueda antes de continuar, y no usar en `calculadora` ninguna
   cifra que no haya aparecido literalmente en un resultado previo de
   `buscar_en_tfm`.

**Resultado tras el arreglo (re-ejecución con los mismos prompts):**

✅ **Confirmado el mecanismo causal.** Tras cambiar el ejemplo del docstring
a cifras genéricas (`"12 + 34 + 56"`), la calculadora pasó a usar
`(12 + 34) + 56 = 102` — el modelo siguió copiando literalmente el
ejemplo del docstring de la tool como si fuera un dato real, solo que
ahora el resultado es obviamente absurdo en vez de casualmente correcto.
Esto confirma sin ambigüedad que el modelo trata el ejemplo de la
descripción de la tool como una plantilla a rellenar, no como
documentación para humanos.

✅ **Mejora real de honestidad.** La respuesta final ahora reconoce
explícitamente la falta de información ("no puedo proporcionar una
respuesta precisa... debido a la falta de información") ANTES de mostrar
el cálculo con datos de ejemplo, y lo presenta con el matiz "si asumimos
que las cantidades mencionadas... son correctas". Es un comportamiento
mucho más seguro que el original, que presentaba el resultado inventado
con total confianza y sin advertencia.

❌ **No solucionado: la instrucción de "una tool a la vez" fue ignorada.**
A pesar de añadirla explícitamente y en primer lugar en el `SYSTEM_PROMPT`,
el modelo siguió emitiendo las 4 tool calls (3 búsquedas + 1 cálculo) en
un único turno, sin esperar ningún resultado intermedio. Esto sugiere que
el comportamiento es más estructural que corregible solo vía prompt:
coincide con lo encontrado en la investigación (Llama 3.1 no tiene
entrenamiento ni soporte formal para tool-calling estrictamente
secuencial vía instrucción de sistema en este contexto).

**Conclusión combinada (causas 2 y 3 interactúan):** el problema de fondo
no es uno solo, son dos que se refuerzan: (a) el retriever sigue sin
encontrar las cifras de canciones por género para estas formulaciones de
pregunta (limitación ya documentada en el punto 2), y (b) el modelo,
al no esperar esos resultados turno a turno, no tiene oportunidad de
"darse cuenta" de que las búsquedas fallaron antes de plantear el cálculo,
y recurre al único número que tiene disponible en el contexto del prompt:
el ejemplo del docstring. El arreglo del docstring evitó que ese fallback
fuera silenciosamente correcto por casualidad, pero no resuelve ni el
problema de recall ni el de no-secuencialidad.

**Próximos pasos razonables (no aplicados, quedan para iterar):**
- Para el recall: las mejoras ya listadas en el punto 2 (overlap, re-ranking).
- Para forzar secuencialidad real: usar `bind_tools(..., tool_choice=...)`
  para restringir explícitamente a una tool por paso, o estructurar el
  agente con LangGraph de forma manual (en vez de `create_agent`
  prefabricado) para controlar el flujo turno a turno en vez de depender
  de que el modelo respete la instrucción por su cuenta.
- Eliminar por completo cualquier número de ejemplo de los docstrings de
  tools que devuelvan datos numéricos, y sustituirlos por descripciones
  puramente verbales del formato esperado, para minimizar el riesgo de
  fuga de ejemplos en cualquier escenario futuro.

## 4. (Espacio para próximas entradas)