"""
src/text_to_sql.py
==================
Pipeline agentico Text-to-SQL con LangGraph.
Nodos: plan -> write -> execute -> fix (retry) -> interpret
Los prompts estan en ingles porque los datos de la BD estan en ingles.
La interpretacion final se da en espanol.
"""

import json
import re
from typing import Any, Optional, TypedDict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.db import get_schema, run_query

MAX_RETRIES = 2


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class QueryPlan(BaseModel):
    reasoning:        str          = Field(description="Step by step reasoning to answer the question")
    tables_needed:    list[str]    = Field(description="List of table names needed for the query")
    join_required:    bool         = Field(description="True if a JOIN between tables is needed")
    aggregation:      bool         = Field(description="True if GROUP BY or aggregation is needed")
    filter_condition: Optional[str] = Field(default=None, description="Filter condition needed, or null")


class SQLQuery(BaseModel):
    sql: str = Field(description="Valid SQLite query ending with semicolon")


class FixedQuery(BaseModel):
    explanation: str = Field(description="What was wrong and what was fixed")
    sql:         str = Field(description="Corrected SQLite query")


# ── Parsers ───────────────────────────────────────────────────────────────────

_plan_parser  = JsonOutputParser(pydantic_object=QueryPlan)
_query_parser = JsonOutputParser(pydantic_object=SQLQuery)
_fix_parser   = JsonOutputParser(pydantic_object=FixedQuery)

# ── Prompts (en ingles para coincidir con los datos) ──────────────────────────

_PLAN_TEMPLATE = PromptTemplate(
    template="""\
You are an expert SQL planner for the Northwind Traders SQLite database.
Given the schema and the user question, produce a structured plan.

Schema:
{schema}

User question: {question}

{format_instructions}

Return ONLY valid JSON, no markdown fences, no extra text.""",
    input_variables=["schema", "question"],
    partial_variables={"format_instructions": _plan_parser.get_format_instructions()},
)

_WRITE_TEMPLATE = PromptTemplate(
    template="""\
You are an expert SQLite query writer for Northwind Traders.

Schema:
{schema}

User question: {question}
Query plan: {plan}

{format_instructions}

CRITICAL RULES:
- Revenue = order_details.quantity * products.price  (NO unit_price column)
- Customer name column is customer_name (NOT company_name)
- Employee full name: first_name || ' ' || last_name
- Use strftime('%Y-%m', order_date) for date grouping
- Use || for string concatenation
- End the query with a semicolon

Return ONLY valid JSON, no markdown fences, no extra text.""",
    input_variables=["schema", "question", "plan"],
    partial_variables={"format_instructions": _query_parser.get_format_instructions()},
)

_INTERPRET_TEMPLATE = ChatPromptTemplate.from_template("""\
You are a data analyst at Northwind Traders.
User question: {question}
SQL executed: {sql}
Results (JSON): {results}

Answer in Spanish, clearly and concisely in 2-4 sentences.
If results are empty, say so and suggest a possible reason.""")

_FIX_TEMPLATE = PromptTemplate(
    template="""\
You are an expert SQLite debugger for Northwind Traders.

Schema:
{schema}

SQL with error:
{sql}

Error message: {error}

{format_instructions}

CRITICAL RULES:
- Revenue = order_details.quantity * products.price  (NO unit_price column)
- Customer name column is customer_name (NOT company_name)
- Use || for string concatenation, NOT CONCAT()
- Use strftime() for date functions, NOT YEAR() or DATE_FORMAT()

Return ONLY valid JSON, no markdown fences, no extra text.""",
    input_variables=["schema", "sql", "error"],
    partial_variables={"format_instructions": _fix_parser.get_format_instructions()},
)


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question:     str
    schema:       str
    doc_context:  str
    plan:         Optional[str]
    sql:          Optional[str]
    results:      Optional[Any]
    error:        Optional[str]
    retries:      int
    final_answer: Optional[str]
    trace:        list[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json(response) -> dict:
    content = getattr(response, "content", str(response)).strip()
    clean   = re.sub(r"```(?:json)?\s*", "", content).replace("```", "").strip()
    match   = re.search(r"\{.*\}", clean, re.DOTALL)
    return json.loads(match.group() if match else clean)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def _plan(state: AgentState, llm: ChatGroq) -> AgentState:
    raw  = llm.invoke(_PLAN_TEMPLATE.format(schema=state["schema"], question=state["question"]))
    plan = _parse_json(raw)
    return {**state,
            "plan":  json.dumps(plan, ensure_ascii=False),
            "trace": state["trace"] + [f"plan: tables={plan.get('tables_needed')}"]}


def _write(state: AgentState, llm: ChatGroq) -> AgentState:
    raw = llm.invoke(_WRITE_TEMPLATE.format(
        schema=state["schema"], question=state["question"], plan=state["plan"]
    ))
    sql = _parse_json(raw).get("sql", "").strip()
    return {**state, "sql": sql, "error": None,
            "trace": state["trace"] + [f"sql: {sql[:100]}"]}


def _execute(state: AgentState) -> AgentState:
    try:
        df      = run_query(state["sql"])
        results = df.to_dict(orient="records")
        return {**state, "results": results, "error": None,
                "trace": state["trace"] + [f"execute: {len(results)} row(s)"]}
    except Exception as exc:
        return {**state, "results": None, "error": str(exc),
                "retries": state["retries"] + 1,
                "trace": state["trace"] + [f"execute error: {str(exc)[:80]}"]}


def _fix(state: AgentState, llm: ChatGroq) -> AgentState:
    raw = llm.invoke(_FIX_TEMPLATE.format(
        schema=state["schema"], sql=state["sql"], error=state["error"]
    ))
    result = _parse_json(raw)
    return {**state, "sql": result.get("sql", "").strip(), "error": None,
            "trace": state["trace"] + [f"fix: {result.get('explanation', '')[:80]}"]}


def _interpret(state: AgentState, llm: ChatGroq) -> AgentState:
    if state["error"] and state["retries"] >= MAX_RETRIES:
        answer = f"No pude generar una query valida despues de {MAX_RETRIES} intentos. Error: {state['error']}"
    else:
        results_str = json.dumps(state["results"] or [], ensure_ascii=False, default=str)
        prompt      = _INTERPRET_TEMPLATE.format_messages(
            question=state["question"],
            sql=state["sql"] or "",
            results=results_str,
        )
        answer = llm.invoke(prompt).content.strip()

    return {**state, "final_answer": answer,
            "trace": state["trace"] + ["interpret: done"]}


def _route(state: AgentState) -> str:
    if state["error"] is None:
        return "interpret"
    return "fix" if state["retries"] < MAX_RETRIES else "interpret"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_sql_graph(llm: ChatGroq):
    builder = StateGraph(AgentState)

    builder.add_node("plan",      lambda s: _plan(s, llm))
    builder.add_node("write",     lambda s: _write(s, llm))
    builder.add_node("execute",   _execute)
    builder.add_node("fix",       lambda s: _fix(s, llm))
    builder.add_node("interpret", lambda s: _interpret(s, llm))

    builder.set_entry_point("plan")
    builder.add_edge("plan",      "write")
    builder.add_edge("write",     "execute")
    builder.add_edge("fix",       "execute")
    builder.add_edge("interpret", END)
    builder.add_conditional_edges("execute", _route, {"interpret": "interpret", "fix": "fix"})

    return builder.compile()


def run_sql_pipeline(question: str, graph, schema: str, doc_context: str = "") -> dict:
    state = graph.invoke({
        "question":     question,
        "schema":       schema,
        "doc_context":  doc_context,
        "plan":         None,
        "sql":          None,
        "results":      None,
        "error":        None,
        "retries":      0,
        "final_answer": None,
        "trace":        [],
    })
    return {
        "final_answer": state["final_answer"],
        "sql":          state["sql"],
        "results":      state["results"],
        "trace":        state["trace"],
    }
