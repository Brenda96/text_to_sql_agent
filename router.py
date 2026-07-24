"""
src/router.py
=============
Clasificador de preguntas del usuario en tres categorias:
  - northwind_doc : respondible con el PDF de promociones de Northwind
  - northwind_sql : requiere consultar la base de datos
  - chitchat      : fuera de dominio, se rechaza
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

CATEGORY_DOC      = "northwind_doc"
CATEGORY_SQL      = "northwind_sql"
CATEGORY_CHITCHAT = "chitchat"

# El prompt esta en ingles porque el PDF y los datos estan en ingles.
# Se incluyen ejemplos reales del contenido del PDF y de la BD
# para que el LLM distinga correctamente entre las dos categorias.
_SYSTEM_PROMPT = """\
You are a question classifier for the Northwind Traders assistant.
Reply ONLY with the exact category name, no other text.

--- CATEGORY: northwind_doc ---
The question can be answered using the HR document (Employee Promotions & Benefits FY2025).
This document contains: sales performance tiers (Gold/Silver/Bronze), salary increases,
one-time bonuses, tenure recognition awards, regional staff development awards,
product knowledge incentive, employee benefits (health, vacation, pension, meal allowance),
eligibility rules, and payment schedule.

Use this category for questions about:
- Promotion criteria, tiers, bonuses, salary increases
- Employee benefits, vacation days, health insurance, parental leave
- Tenure awards, years of service recognition
- Regional development awards
- Product knowledge certification and incentives
- Specific employees mentioned in the document (Peacock, Leverling, Davolio, King, Fuller, Buchanan)
- Payment dates and schedules

Examples:
- "What are the Gold tier requirements?"
- "How much bonus does a Silver employee get?"
- "What benefits does Northwind offer?"
- "Who received the top performer award?"
- "What is the parental leave policy?"
- "When are bonuses paid?"
- "Cuales son los niveles de incentivos?" (Spanish is also valid)
- "Que bono recibe el nivel Gold?"

--- CATEGORY: northwind_sql ---
The question requires querying live data from the Northwind database.
The database contains: categories, customers, employees, order_details, orders, products, shippers, suppliers.

Use this category for questions about:
- Counts, totals, rankings, statistics from the database
- Specific orders, customers, products by name
- Revenue calculations, sales volumes
- Which shipper, supplier, or customer has the most/least of something
- Product prices, inventory, categories from the catalog

Examples:
- "How many orders does QUICK-Stop have?"
- "What is the top selling product by revenue?"
- "How many customers per country?"
- "Which employee processed the most orders?"
- "What products are in the Beverages category?"
- "What is the price of Chai?"
- "Cuantos pedidos tiene QUICK-Stop?"
- "Cual es el producto mas vendido?"

--- CATEGORY: chitchat ---
The question is not related to Northwind Traders at all.

Examples:
- "How do I cook pasta?"
- "Who won the World Cup?"
- "What is the capital of France?"
- "Hello, how are you?"
- "Tell me a joke"

Reply with ONLY one of these exact strings: northwind_doc | northwind_sql | chitchat
"""


def classify(question: str, llm: ChatGroq) -> str:
    """
    Clasifica la pregunta del usuario.

    Args:
        question: Pregunta en lenguaje natural (ingles o espanol).
        llm:      Instancia de ChatGroq configurada para el router.

    Returns:
        Nombre de la categoria: 'northwind_doc', 'northwind_sql' o 'chitchat'.
    """
    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ])
    raw = response.content.strip().lower()

    if CATEGORY_SQL in raw:
        return CATEGORY_SQL
    if CATEGORY_DOC in raw:
        return CATEGORY_DOC
    if CATEGORY_CHITCHAT in raw:
        return CATEGORY_CHITCHAT
    return CATEGORY_DOC  # fallback conservador
