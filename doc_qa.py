"""
src/doc_qa.py
=============
Pipeline QA basado en el PDF de Northwind Traders.
El texto completo del PDF se inyecta en el prompt del sistema.
El documento es el reporte de Promociones y Beneficios FY2025.
"""
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pypdf import PdfReader

_SYSTEM_PROMPT = """\
You are the virtual assistant for Northwind Traders, a company specialized in
importing and exporting fine foods and specialty beverages.

You have access to the following internal HR document:
Employee Promotions & Benefits Program — FY2025

Document content:
======================
{context}
======================

Instructions:
- Answer ONLY based on information in the document above.
- If the information is not in the document, say clearly:
  "This information is not available in the Northwind document."
- Always answer in Spanish, in a precise and professional tone.
- Quote specific numbers, dates, and names from the document when relevant.
"""


def load_pdf(pdf_path: str) -> str:
    """
    Extrae el texto completo del PDF.

    Args:
        pdf_path: Ruta al archivo PDF.

    Returns:
        Texto extraido como string.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    reader = PdfReader(str(path))
    pages  = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def build_qa_chain(context: str, llm: ChatGroq):
    """
    Construye la chain QA con el contexto del PDF inyectado.

    Args:
        context: Texto completo del PDF.
        llm:     Instancia de ChatGroq.

    Returns:
        Chain lista para invocar con {"question": ...}.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT.format(context=context)),
        ("human",  "{question}"),
    ])
    return prompt | llm | StrOutputParser()


def answer_from_doc(question: str, chain) -> str:
    return chain.invoke({"question": question})
