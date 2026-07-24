# 🤖 Northwind Traders — Asistente Inteligente

Chatbot con enrutamiento inteligente que responde preguntas sobre la empresa ficticia **Northwind Traders**, combinando dos fuentes de información distintas: un documento de RRHH y una base de datos SQL en vivo. Usa **Groq** como proveedor de LLM y **LangGraph** para orquestar un pipeline agéntico de Text-to-SQL con auto-corrección.

---

## 🧠 Cómo funciona

Cada mensaje del usuario pasa primero por un **router** (clasificador) que decide a cuál de tres flujos enviarlo:

```
Pregunta del usuario
        │
        ▼
 ┌─────────────┐
 │   Router     │   LLM ligero (llama-3.1-8b-instant)
 │ (clasifica)  │   clasifica en 1 de 3 categorías
 └─────────────┘
        │
   ┌────┼────────────────┐
   ▼    ▼                ▼
 Doc   SQL           Chitchat
  QA   (agéntico)    (fuera de dominio)
```

### 1. `northwind_doc` — Preguntas sobre RRHH
Se responden usando el texto completo de un PDF interno (**Employee Promotions & Benefits FY2025**) inyectado directamente en el prompt del sistema. Cubre niveles de bonos (Gold/Silver/Bronze), beneficios, licencias, reconocimientos por antigüedad, etc.

### 2. `northwind_sql` — Preguntas sobre datos de negocio
Se resuelven con un **grafo agéntico construido en LangGraph** que convierte lenguaje natural en SQL, lo ejecuta, y se auto-corrige si falla:

```
plan → write → execute ──✅──▶ interpret
                 │
                 ❌ error
                 ▼
                fix ──▶ execute (reintenta, máx. 2 veces)
```

- **plan**: razona qué tablas y joins necesita la pregunta
- **write**: genera la query SQLite siguiendo reglas estrictas del esquema (evita alucinar columnas)
- **execute**: corre la query contra la base SQLite
- **fix**: si la query falla, un LLM analiza el error y la corrige (hasta 2 reintentos)
- **interpret**: traduce el resultado tabular a una respuesta en español, natural y concisa

Cada paso queda registrado en un **trace** que se muestra en la interfaz, para que puedas ver exactamente qué razonó el agente y qué SQL generó.

### 3. `chitchat` — Fuera de dominio
Cualquier pregunta no relacionada con Northwind (ej. "cuéntame un chiste") se rechaza educadamente, sin gastar tokens en los otros pipelines.

---

## ⚙️ Stack técnico

| Componente | Herramienta |
|---|---|
| Interfaz | Gradio (Blocks, tema oscuro personalizado) |
| LLM | Groq — `llama-3.3-70b-versatile` (respuestas) + `llama-3.1-8b-instant` (router) |
| Orquestación agéntica | LangGraph (StateGraph) |
| Validación de salidas del LLM | Pydantic + JsonOutputParser de LangChain |
| Base de datos | SQLite vía SQLAlchemy |
| Lectura de PDF | pypdf |
| Variables de entorno | python-dotenv |

---

## 📁 Estructura del proyecto

```
northwind-chatbot/
├── app.py                 # Interfaz Gradio (chat + panel de trace)
├── requirements.txt
├── Dockerfile              # (opcional) despliegue vía contenedor
├── .env.example            # Plantilla de variables de entorno
├── src/
│   ├── __init__.py
│   ├── router.py           # Clasificador de intención (3 categorías)
│   ├── doc_qa.py            # Pipeline QA sobre el PDF de RRHH
│   ├── text_to_sql.py        # Grafo agéntico LangGraph (plan/write/execute/fix/interpret)
│   ├── db.py                 # Conexión SQLite + esquema hardcodeado de Northwind
│   └── pipeline.py            # Orquestador: llama al router y despacha al flujo correcto
├── northwind.db              # Base de datos SQLite (no incluida por tamaño/privacidad)
└── northwind_info.pdf         # Documento de RRHH (no incluido por privacidad)
```

> ⚠️ `northwind.db` y `northwind_info.pdf` no están en el repositorio. Debes colocarlos tú mismo en la raíz del proyecto antes de ejecutar la app (ver sección de instalación).

---

## 🚀 Instalación y uso local

```bash
git clone https://github.com/TU-USUARIO/northwind-chatbot.git
cd northwind-chatbot

python -m venv venv
source venv/bin/activate       # En Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configurar variables de entorno

```bash
cp .env.example .env
```

Edita `.env` y coloca tu API key de Groq (consíguela gratis en https://console.groq.com):

```
GROQ_API_KEY=tu_api_key_aqui
DB_PATH=northwind.db
PDF_PATH=northwind_info.pdf
GROQ_MODEL=llama-3.3-70b-versatile
```

### Agregar los datos

Coloca en la raíz del proyecto:
- `northwind.db` — base SQLite con el esquema de Northwind (categories, customers, employees, orders, order_details, products, shippers, suppliers)
- `northwind_info.pdf` — el documento de beneficios/promociones de RRHH

### Ejecutar

```bash
python app.py
```

La app abrirá en `http://localhost:7860`.

---

## 💬 Ejemplos de preguntas

**Sobre RRHH (documento):**
- "¿Cuáles son los requisitos del nivel Gold?"
- "¿Qué beneficios ofrece Northwind?"
- "¿Cuál es la política de licencia parental?"

**Sobre la base de datos (SQL en vivo):**
- "¿Cuál es el producto más vendido por ingresos?"
- "¿Cuántos clientes hay por país?"
- "¿Qué empleado procesó más pedidos?"

---

## 👤 Autora

Hecho por **Brenda Jauregui**.
