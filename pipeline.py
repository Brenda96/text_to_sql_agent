"""
src/pipeline.py
===============
Orquestador principal del chatbot Northwind.
Inicializa los componentes y despacha cada pregunta al pipeline correcto.
"""
import os

from langchain_groq import ChatGroq

from src.db import get_schema
from src.doc_qa import build_qa_chain, load_pdf
from src.router import CATEGORY_CHITCHAT, CATEGORY_DOC, CATEGORY_SQL, classify
from src.text_to_sql import build_sql_graph, run_sql_pipeline

_CHITCHAT_MSG = (
    "Solo puedo responder preguntas relacionadas con Northwind Traders: "
    "productos, empleados, politicas comerciales, pedidos y operaciones de la empresa. "
    "Tienes alguna pregunta sobre Northwind?"
)


def init_pipeline(
    db_path:  str = "northwind.db",
    pdf_path: str = "northwind_info.pdf",
    model:    str = "llama-3.3-70b-versatile",
) -> dict:
    """
    Inicializa todos los componentes del pipeline.
    Se llama una sola vez al arrancar la aplicacion.

    Args:
        db_path:  Ruta a la base de datos SQLite de Northwind.
        pdf_path: Ruta al PDF de Northwind.
        model:    Nombre del modelo de Groq.

    Returns:
        Diccionario con los componentes listos para usar.

    Raises:
        ValueError: Si GROQ_API_KEY no esta configurada.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY no configurada. Crea un archivo .env o configura la variable.")

    # LLM principal (QA y SQL)
    llm_main = ChatGroq(model=model, temperature=0, api_key=api_key)

    # LLM del router (solo clasifica, pocas tokens)
    llm_router = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key, max_tokens=20)

    # Esquema de la base de datos
    schema = get_schema(db_path)

    # Contexto del PDF
    try:
        pdf_context = load_pdf(pdf_path)
    except FileNotFoundError:
        pdf_context = ""

    # Chain QA del PDF
    qa_chain = build_qa_chain(pdf_context, llm_main)

    # Grafo Text-to-SQL
    sql_graph = build_sql_graph(llm_main)

    return {
        "llm_router":   llm_router,
        "qa_chain":     qa_chain,
        "sql_graph":    sql_graph,
        "schema":       schema,
        "pdf_context":  pdf_context,
        "db_path":      db_path,
    }


def process(question: str, components: dict) -> dict:
    """
    Procesa una pregunta del usuario a traves del pipeline completo.

    Args:
        question:   Pregunta del usuario.
        components: Diccionario retornado por init_pipeline().

    Returns:
        Dict con keys: final_answer, categoria, sql, results, trace.
    """
    categoria = classify(question, components["llm_router"])

    if categoria == CATEGORY_DOC:
        from src.doc_qa import answer_from_doc
        answer = answer_from_doc(question, components["qa_chain"])
        return {"final_answer": answer, "categoria": categoria, "sql": None, "results": None, "trace": []}

    if categoria == CATEGORY_SQL:
        result = run_sql_pipeline(
            question=question,
            graph=components["sql_graph"],
            schema=components["schema"],
            doc_context=components["pdf_context"],
        )
        result["categoria"] = categoria
        return result

    # chitchat
    return {"final_answer": _CHITCHAT_MSG, "categoria": categoria, "sql": None, "results": None, "trace": []}
