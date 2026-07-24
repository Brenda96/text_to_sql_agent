"""
app.py
======
Interfaz Gradio para el chatbot Northwind Traders.
Disenada para despliegue en Hugging Face Spaces.
"""

import os
import gradio as gr
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.pipeline import init_pipeline, process
from src.router import CATEGORY_CHITCHAT, CATEGORY_DOC, CATEGORY_SQL

# ── Inicializacion ─────────────────────────────────────────────────────────────
DB_PATH  = os.getenv("DB_PATH",    "northwind.db")
PDF_PATH = os.getenv("PDF_PATH",   "northwind_info.pdf")
MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

try:
    _components = init_pipeline(db_path=DB_PATH, pdf_path=PDF_PATH, model=MODEL)
    _ready      = True
    _init_error = None
except Exception as exc:
    _components = None
    _ready      = False
    _init_error = str(exc)

_CATEGORY_LABELS = {
    CATEGORY_DOC:      "Document",
    CATEGORY_SQL:      "Database",
    CATEGORY_CHITCHAT: "Out of scope",
}

_EXAMPLES_DOC = [
    "What are the Gold tier requirements?",
    "How much bonus does a Silver employee get?",
    "What benefits does Northwind offer?",
    "Who received the top performer award in FY2024?",
    "What is the parental leave policy?",
]
_EXAMPLES_SQL = [
    "What is the top selling product by revenue?",
    "How many customers per country? (top 10)",
    "Which employee processed the most orders?",
    "How many orders does each shipper handle?",
    "What are the most expensive products?",
]
_EXAMPLES_CHITCHAT = ["Hello, how are you?"]

_TRACE_LABELS = {
    "plan":      "Planning",
    "sql":       "SQL Generation",
    "execute":   "Execution",
    "fix":       "Fix",
    "interpret": "Interpretation",
}
_TRACE_ICONS = {
    "plan":      "📋",
    "sql":       "✍️",
    "execute":   "⚡",
    "fix":       "🔧",
    "interpret": "💬",
}
_TRACE_EMPTY = "*Send a message to see the pipeline trace.*"

# ── CSS ─────────────────────────────────────────────────────────────────────────
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

/* ── Fondo general ── */
.gradio-container { background: #0f1117 !important; }
body { background: #0f1117 !important; }

/* ── Header ── */
#nw-header {
    background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 50%, #52b788 100%);
    border-radius: 14px;
    padding: 28px 36px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
#nw-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
#nw-header::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 30%;
    width: 260px; height: 260px;
    background: rgba(255,255,255,0.03);
    border-radius: 50%;
}
#nw-header h1 {
    color: #ffffff !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.5px;
    line-height: 1.2 !important;
}
#nw-header p {
    color: #95d5b2 !important;
    font-size: 0.92rem !important;
    margin: 0 !important;
    font-weight: 400;
}

/* ── Sidebar ── */
#sidebar {
    background: #161b27 !important;
    border: 1px solid #2a3347 !important;
    border-radius: 12px !important;
    padding: 20px 14px !important;
}
.sidebar-section-doc {
    color: #52b788 !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin: 0 0 10px 0 !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid #2d6a4f !important;
}
.sidebar-section-sql {
    color: #74b9ff !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin: 18px 0 10px 0 !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid #2c4a6e !important;
}
.sidebar-section-other {
    color: #a0aec0 !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin: 18px 0 10px 0 !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid #2a3347 !important;
}
.sidebar-legend {
    font-size: 0.75rem !important;
    color: #718096 !important;
    margin-top: 18px !important;
    padding-top: 14px !important;
    border-top: 1px solid #2a3347 !important;
    line-height: 1.9 !important;
}

/* ── Botones ejemplo ── */
.btn-doc button {
    background: rgba(82, 183, 136, 0.12) !important;
    color: #74c69d !important;
    border: 1px solid rgba(82, 183, 136, 0.25) !important;
    border-radius: 7px !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    padding: 6px 10px !important;
    width: 100% !important;
    text-align: left !important;
    margin-bottom: 4px !important;
    transition: all 0.15s ease !important;
    line-height: 1.4 !important;
}
.btn-doc button:hover {
    background: rgba(82, 183, 136, 0.22) !important;
    border-color: rgba(82, 183, 136, 0.5) !important;
    color: #95d5b2 !important;
}
.btn-sql button {
    background: rgba(116, 185, 255, 0.1) !important;
    color: #74b9ff !important;
    border: 1px solid rgba(116, 185, 255, 0.22) !important;
    border-radius: 7px !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    padding: 6px 10px !important;
    width: 100% !important;
    text-align: left !important;
    margin-bottom: 4px !important;
    transition: all 0.15s ease !important;
    line-height: 1.4 !important;
}
.btn-sql button:hover {
    background: rgba(116, 185, 255, 0.2) !important;
    border-color: rgba(116, 185, 255, 0.45) !important;
    color: #a8d8ff !important;
}
.btn-other button {
    background: rgba(160, 174, 192, 0.1) !important;
    color: #a0aec0 !important;
    border: 1px solid rgba(160, 174, 192, 0.2) !important;
    border-radius: 7px !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    padding: 6px 10px !important;
    width: 100% !important;
    text-align: left !important;
    margin-bottom: 4px !important;
    transition: all 0.15s ease !important;
    line-height: 1.4 !important;
}
.btn-other button:hover {
    background: rgba(160, 174, 192, 0.2) !important;
    color: #cbd5e0 !important;
}

/* ── Chat panel ── */
#chat-col {
    background: #161b27 !important;
    border: 1px solid #2a3347 !important;
    border-radius: 12px !important;
    padding: 0 !important;
    overflow: hidden !important;
}
#chatbot {
    height: 490px !important;
    background: #161b27 !important;
    border: none !important;
    border-bottom: 1px solid #2a3347 !important;
}
/* Burbujas usuario */
#chatbot .message.user {
    background: #1e3a5f !important;
    color: #e2e8f0 !important;
    border-radius: 12px 12px 4px 12px !important;
}
/* Burbujas bot */
#chatbot .message.bot {
    background: #1e2535 !important;
    color: #e2e8f0 !important;
    border-radius: 12px 12px 12px 4px !important;
    border-left: 3px solid #2d6a4f !important;
}

/* ── Input area ── */
#input-area {
    background: #161b27 !important;
    padding: 12px 14px !important;
}
#txt-input textarea {
    background: #0f1117 !important;
    color: #e2e8f0 !important;
    border: 1px solid #2a3347 !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
    padding: 10px 14px !important;
}
#txt-input textarea:focus {
    border-color: #2d6a4f !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(45, 106, 79, 0.25) !important;
}
#txt-input textarea::placeholder { color: #4a5568 !important; }

#btn-send button {
    background: linear-gradient(135deg, #2d6a4f, #40916c) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    height: 44px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(45, 106, 79, 0.35) !important;
}
#btn-send button:hover {
    background: linear-gradient(135deg, #40916c, #52b788) !important;
    box-shadow: 0 4px 12px rgba(45, 106, 79, 0.5) !important;
    transform: translateY(-1px) !important;
}

#btn-clear button {
    background: transparent !important;
    color: #4a5568 !important;
    border: 1px solid #2a3347 !important;
    border-radius: 7px !important;
    font-size: 0.78rem !important;
    transition: all 0.15s ease !important;
    margin-top: 6px !important;
}
#btn-clear button:hover {
    color: #718096 !important;
    border-color: #3d4f6e !important;
}

/* ── Trace panel ── */
#trace-col {
    background: #161b27 !important;
    border: 1px solid #2a3347 !important;
    border-radius: 12px !important;
    padding: 20px 18px !important;
    min-height: 580px !important;
}
.trace-header {
    color: #a0aec0 !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin: 0 0 14px 0 !important;
    padding-bottom: 10px !important;
    border-bottom: 1px solid #2a3347 !important;
}
#trace-content {
    font-size: 0.82rem !important;
    line-height: 1.75 !important;
    color: #cbd5e0 !important;
}
#trace-content strong { color: #e2e8f0 !important; font-weight: 600 !important; }
#trace-content code {
    background: #0f1117 !important;
    color: #74c69d !important;
    border: 1px solid #2a3347 !important;
    border-radius: 4px !important;
    padding: 1px 6px !important;
    font-size: 0.76rem !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}
#trace-content pre {
    background: #0f1117 !important;
    border: 1px solid #2a3347 !important;
    border-radius: 8px !important;
    padding: 12px 14px !important;
    overflow-x: auto !important;
}
#trace-content pre code {
    background: transparent !important;
    border: none !important;
    color: #74b9ff !important;
    font-size: 0.78rem !important;
    padding: 0 !important;
}
#trace-content hr {
    border-color: #2a3347 !important;
    margin: 14px 0 !important;
}
/* Separacion entre pasos de la traza */
#trace-content p {
    margin: 0 0 4px 0 !important;
    line-height: 1.6 !important;
}
/* El detalle del paso en su propia linea como pill */
#trace-content p > code {
    display: inline-block !important;
    max-width: 100% !important;
    overflow-x: auto !important;
    white-space: pre-wrap !important;
    word-break: break-all !important;
    margin-top: 3px !important;
    padding: 4px 10px !important;
    font-size: 0.74rem !important;
    line-height: 1.5 !important;
}

footer { display: none !important; }
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def _results_to_markdown(results: list | None) -> str:
    if not results:
        return ""
    df = pd.DataFrame(results)
    return "\n\n**Results:**\n\n" + df.to_markdown(index=False)


def _build_trace_md(trace: list[str], categoria: str, sql: str | None) -> str:
    badge = {
        CATEGORY_DOC:      ("🟢", "Document"),
        CATEGORY_SQL:      ("🔵", "Database"),
        CATEGORY_CHITCHAT: ("⚪", "Out of scope"),
    }.get(categoria, ("⚪", categoria))

    # Cada bloque separado por linea en blanco para que Markdown
    # los renderice como parrafos independientes y no en linea
    lines = [f"{badge[0]} **Category:** {badge[1]}", ""]

    if trace:
        lines.append("**Pipeline steps:**")
        lines.append("")
        for step in trace:
            key, _, detail = step.partition(":")
            key        = key.strip()
            detail     = detail.strip()
            step_label = _TRACE_LABELS.get(key, key.capitalize())
            icon       = _TRACE_ICONS.get(key, "•")
            # Doble salto de linea entre cada paso para que no se peguen
            if detail:
                lines.append(f"{icon} **{step_label}**")
                lines.append(f"`{detail}`")
                lines.append("")
            else:
                lines.append(f"{icon} **{step_label}**")
                lines.append("")
    else:
        lines.append("*This category does not use the agentic pipeline.*")
        lines.append("")

    if sql:
        lines += ["---", "", "**Generated SQL:**", "", f"```sql\n{sql}\n```"]

    return "\n".join(lines)


# ── Chat logic ─────────────────────────────────────────────────────────────────

def chat(message: str, history: list) -> tuple[str, list, str]:
    if not _ready:
        return "", history + [[message, f"Error: {_init_error}"]], _TRACE_EMPTY
    if not message.strip():
        return "", history, _TRACE_EMPTY

    result    = process(message, _components)
    answer    = result.get("final_answer", "No response.")
    categoria = result.get("categoria", "")
    sql       = result.get("sql")
    results   = result.get("results")
    trace     = result.get("trace", [])

    label    = _CATEGORY_LABELS.get(categoria, categoria)
    response = f"**[{label}]**\n\n{answer}"
    if results:
        response += _results_to_markdown(results)

    return "", history + [[message, response]], _build_trace_md(trace, categoria, sql)


def clear_all() -> tuple[list, str, str]:
    return [], "", _TRACE_EMPTY


# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Northwind Traders — Assistant",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ),
    css=_CSS,
) as demo:

    # ── Header ─────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="nw-header">
        <h1>Northwind Traders</h1>
        <p>Intelligent Assistant &mdash; Ask about HR policies or query the sales database in real time.</p>
    </div>
    """)

    if not _ready:
        gr.Warning(f"Pipeline unavailable: {_init_error}")

    with gr.Row(equal_height=False, variant="panel"):

        # ── Sidebar ────────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=200, elem_id="sidebar"):
            gr.HTML('<div class="sidebar-section-doc">HR Document</div>')
            doc_btns = [
                gr.Button(ex, size="sm", elem_classes="btn-doc")
                for ex in _EXAMPLES_DOC
            ]
            gr.HTML('<div class="sidebar-section-sql">Database</div>')
            sql_btns = [
                gr.Button(ex, size="sm", elem_classes="btn-sql")
                for ex in _EXAMPLES_SQL
            ]
            gr.HTML('<div class="sidebar-section-other">Other</div>')
            other_btns = [
                gr.Button(ex, size="sm", elem_classes="btn-other")
                for ex in _EXAMPLES_CHITCHAT
            ]
            gr.HTML("""
            <div class="sidebar-legend">
                🟢 <strong>Document</strong> &mdash; HR PDF<br>
                🔵 <strong>Database</strong> &mdash; live SQL<br>
                ⚪ <strong>Out of scope</strong> &mdash; rejected
            </div>
            """)

        # ── Chat ───────────────────────────────────────────────────────────────
        with gr.Column(scale=3, elem_id="chat-col"):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                show_label=False,
                bubble_full_width=False,
                render_markdown=True,
            )
            with gr.Row(elem_id="input-area"):
                txt_input = gr.Textbox(
                    placeholder="Ask about Northwind...",
                    show_label=False,
                    scale=5,
                    autofocus=True,
                    container=False,
                    elem_id="txt-input",
                )
                btn_send = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                    elem_id="btn-send",
                )
            btn_clear = gr.Button(
                "Clear conversation",
                size="sm",
                elem_id="btn-clear",
            )

        # ── Trace ──────────────────────────────────────────────────────────────
        with gr.Column(scale=2, elem_id="trace-col"):
            gr.HTML('<div class="trace-header">Pipeline Trace</div>')
            trace_box = gr.Markdown(
                value=_TRACE_EMPTY,
                elem_id="trace-content",
            )

    # ── Handlers ───────────────────────────────────────────────────────────────
    _outputs = [txt_input, chatbot, trace_box]
    txt_input.submit(chat, [txt_input, chatbot], _outputs)
    btn_send.click(chat,   [txt_input, chatbot], _outputs)
    btn_clear.click(clear_all, outputs=[chatbot, txt_input, trace_box])

    all_btns = doc_btns + sql_btns + other_btns
    all_exs  = _EXAMPLES_DOC + _EXAMPLES_SQL + _EXAMPLES_CHITCHAT
    for btn, ex in zip(all_btns, all_exs):
        btn.click(fn=lambda _ex=ex: _ex, outputs=txt_input).then(
            fn=chat, inputs=[txt_input, chatbot], outputs=_outputs
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
