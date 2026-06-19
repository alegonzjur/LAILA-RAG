"""
agent.py

Agente sobre el TFM de LAILA, con dos herramientas (tools):

1. `buscar_en_tfm`: envuelve la RAG chain (src/rag_chain.py) como tool.
   Úsala para cualquier pregunta sobre el CONTENIDO del documento.
2. `calculadora`: evalúa expresiones aritméticas simples de forma segura
   (sin usar eval()). Útil cuando hay que combinar/operar sobre datos
   numéricos ya recuperados (p. ej. sumar canciones de varios géneros).

Diferencia clave frente a rag_chain.py (chain "pura"):
- La chain de rag_chain.py sigue SIEMPRE la misma secuencia fija:
  retriever -> prompt -> llm. No importa la pregunta, el camino es el mismo.
- El agente usa el LLM para DECIDIR, en tiempo de ejecución y en función
  de la pregunta, qué tool(s) llamar, en qué orden, y si necesita encadenar
  varias llamadas antes de dar la respuesta final.

Ejemplo que ilustra por qué hace falta un agente y no basta una chain:
  "¿Cuántas canciones había en total entre rock, pop y rap antes del split?"
  Esto requiere: (1) recuperar tres cifras distintas del documento,
  (2) sumarlas con precisión. Una chain de RAG normal no puede encadenar
  "buscar tres datos y luego operar sobre ellos" de forma fiable; el agente
  sí, porque puede invocar la tool de búsqueda varias veces y luego la
  calculadora sobre los resultados.

NOTA SOBRE LA VERSIÓN DE LANGCHAIN:
Este archivo usa `langchain.agents.create_agent`, la API estable a partir
de langchain >= 1.0. Las versiones anteriores de este proyecto usaban
`AgentExecutor` + `create_react_agent` con un prompt ReAct escrito a mano
(formato Thought/Action/Observation); esa API se eliminó del paquete
`langchain` en la v1.x (movida, en parte, a `langchain-classic`).
`create_agent` está construida sobre LangGraph internamente y ya no
necesita que definamos el prompt ReAct nosotros mismos: el bucle
"pensar -> llamar tool -> observar -> repetir" lo gestiona la librería.

Ejecutar como demo manual:
    python -m src.agent
"""

import ast
import operator

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

from src import config
from src.rag_chain import build_rag_chain


# --- Tool 1: RAG sobre el TFM ---

_rag_chain = None  # se construye una sola vez (lazy init) y se reutiliza


def _get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = build_rag_chain()
    return _rag_chain


@tool
def buscar_en_tfm(pregunta: str) -> str:
    """
    Busca información dentro del TFM de LAILA (Generación Automática de
    Canciones mediante Modelos de Lenguaje Natural y Redes Neuronales).
    Úsala para cualquier pregunta sobre el contenido del documento:
    metodología, hiperparámetros, datasets, hardware, software, resultados, etc.
    El argumento debe ser una pregunta clara y autocontenida en español.
    """
    chain = _get_rag_chain()
    result = chain.invoke(pregunta)
    return result["answer"]


# --- Tool 2: calculadora segura (sin eval) ---

_OPERADORES_PERMITIDOS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _evaluar_nodo(nodo):
    """Evalúa recursivamente un nodo del AST, permitiendo solo aritmética básica."""
    if isinstance(nodo, ast.Constant):
        if isinstance(nodo.value, (int, float)):
            return nodo.value
        raise ValueError("Solo se permiten números.")
    if isinstance(nodo, ast.BinOp):
        op_type = type(nodo.op)
        if op_type not in _OPERADORES_PERMITIDOS:
            raise ValueError(f"Operador no permitido: {op_type.__name__}")
        izquierda = _evaluar_nodo(nodo.left)
        derecha = _evaluar_nodo(nodo.right)
        return _OPERADORES_PERMITIDOS[op_type](izquierda, derecha)
    if isinstance(nodo, ast.UnaryOp):
        op_type = type(nodo.op)
        if op_type not in _OPERADORES_PERMITIDOS:
            raise ValueError(f"Operador unario no permitido: {op_type.__name__}")
        return _OPERADORES_PERMITIDOS[op_type](_evaluar_nodo(nodo.operand))
    raise ValueError(f"Expresión no permitida: {type(nodo).__name__}")


@tool
def calculadora(expresion: str) -> str:
    """
    Evalúa una expresión aritmética simple (suma, resta, multiplicación,
    división, potencias) y devuelve el resultado numérico.
    Úsala SOLO para operar sobre cifras que ya hayas obtenido como resultado
    real de buscar_en_tfm en este mismo turno de conversación. No utilices
    nunca cifras de ejemplo ni cifras que no hayas visto en una Observation.
    El argumento debe ser una expresión matemática con números, ej: "12 + 34 + 56"
    No evalúes texto, solo expresiones numéricas.
    """
    try:
        arbol = ast.parse(expresion, mode="eval")
        resultado = _evaluar_nodo(arbol.body)
        return str(resultado)
    except Exception as e:
        return f"Error al evaluar la expresión '{expresion}': {e}"


# --- Construcción del agente (API nueva: langchain.agents.create_agent) ---

SYSTEM_PROMPT = """Eres un asistente que responde preguntas sobre el TFM de \
LAILA (Generación Automática de Canciones mediante Modelos de Lenguaje \
Natural y Redes Neuronales).

Tienes dos herramientas:
- buscar_en_tfm: para cualquier pregunta sobre el contenido del documento.
- calculadora: para operar con precisión sobre cifras (sumas, restas, etc.)
  que hayas obtenido PREVIAMENTE como resultado real de buscar_en_tfm.

Reglas importantes:
1. Llama a las herramientas DE UNA EN UNA, nunca varias a la vez. Espera
   siempre el resultado de una herramienta antes de decidir el siguiente paso.
2. Si necesitas varios datos (por ejemplo, cifras de varias secciones),
   busca cada uno por separado con buscar_en_tfm, uno detrás de otro, y
   solo cuando ya tengas TODOS los datos reales, usa calculadora para operar
   sobre ellos.
3. Nunca uses en calculadora una cifra que no hayas visto literalmente en
   el resultado de una llamada anterior a buscar_en_tfm. Si una búsqueda no
   encuentra el dato, dilo explícitamente en tu respuesta final en vez de
   inventar o asumir un número.
4. Responde siempre en español, de forma clara y concisa."""


def build_agent():
    """
    Construye el agente con la API nueva de LangChain (>= 1.0).

    create_agent() devuelve un grafo (LangGraph) ya compilado, que se invoca
    con un dict {"messages": [...]} en formato de chat, no con {"input": ...}
    como en la API antigua de AgentExecutor.
    """
    llm = ChatOllama(model=config.AGENT_MODEL, temperature=0.1)
    tools = [buscar_en_tfm, calculadora]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


def run_agent(agent, pregunta: str) -> str:
    """
    Invoca el agente con una pregunta y devuelve solo el texto de la
    respuesta final (el último mensaje de la conversación).
    """
    result = agent.invoke({"messages": [{"role": "user", "content": pregunta}]})
    mensajes = result["messages"]

    # Útil para depurar: imprime el rastro completo de tool calls,
    # equivalente al verbose=True de la API antigua.
    print("\n--- Rastro de mensajes (razonamiento + tool calls) ---")
    for m in mensajes:
        tipo = m.__class__.__name__
        contenido = getattr(m, "content", "")
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            print(f"[{tipo}] tool_calls={tool_calls}")
        elif contenido:
            print(f"[{tipo}] {contenido[:300]}")
    print("--------------------------------------------------------\n")

    return mensajes[-1].content


def main():
    agent = build_agent()

    preguntas_demo = [
        # Esta requiere SOLO la tool de RAG
        "¿Qué tarjeta gráfica se usó para entrenar los modelos?",
        # Esta requiere encadenar RAG (varias búsquedas) + calculadora
        "¿Cuántas canciones había en total entre rock, pop y rap antes de "
        "dividir en train/test? Busca cada cifra por separado y súmalas.",
    ]

    for pregunta in preguntas_demo:
        print(f"\n{'='*70}\nPREGUNTA: {pregunta}\n{'='*70}")
        respuesta = run_agent(agent, pregunta)
        print(f"RESPUESTA FINAL: {respuesta}")


if __name__ == "__main__":
    main()