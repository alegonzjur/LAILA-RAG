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

## 2. (Espacio para próximas entradas)