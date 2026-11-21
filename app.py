import os
import re
from pathlib import Path
import streamlit as st
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.tools import tool
from langchain.agents import create_agent

st.set_page_config(page_title="FoodHub Order Support Chatbot", page_icon="🍔")
st.title("🍔 FoodHub AI Order Support Chatbot")

# 1. Load Groq API key from Streamlit Secrets or environment variable
api_key = os.environ.get("GROQ_API_KEY", "")
if not api_key:
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        api_key = ""

if not api_key:
    st.error("GROQ_API_KEY not found. Add it under App Settings -> Secrets in Streamlit Cloud.")
    st.stop()
os.environ["GROQ_API_KEY"] = api_key

# 2. Input & Output Guardrails
BULK_PATTERNS = [
    r"\b(all|every)\s+(the\s+)?orders?\b",
    r"\b(all|every)\s+(the\s+)?customers?\b",
    r"\bdump\s+(the\s+)?database\b",
    r"\bentire\s+database\b",
]
INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+)?(previous|prior)\s+instructions\b",
    r"\breveal\s+(the\s+)?(system prompt|credentials|api key)\b",
]

def input_guardrail(message: str):
    s = message.lower()
    if "hacker" in s or any(re.search(p, s) for p in BULK_PATTERNS):
        return False, "I cannot provide bulk or unauthorized order information. I can only assist with your verified order."
    if any(re.search(p, s) for p in INJECTION_PATTERNS):
        return False, "I cannot process instructions that attempt to bypass security controls."
    return True, ""

def output_guardrail(text: str) -> str:
    text = re.sub(r"(?i)(api[_ -]?key|password|secret)\s*[:=]\s*\S+", "[REDACTED]", str(text))
    return text[:1500]

# 3. Cached Chat Agent Initialization
@st.cache_resource
def load_agent(db_path: str):
    if not Path(db_path).exists():
        st.error(f"Database file '{db_path}' not found in the repository.")
        st.stop()
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0, max_tokens=512, max_retries=2)
    db_uri = f"sqlite:///file:{Path(db_path).resolve().as_posix()}?mode=ro&uri=true"
    engine = create_engine(db_uri, connect_args={"check_same_thread": False})
    sql_db = SQLDatabase(engine, include_tables=["orders"], sample_rows_in_table_info=2)
    sql_agent = create_sql_agent(llm=llm, db=sql_db, agent_type="tool-calling", top_k=5, max_iterations=6)

    @tool
    def order_query_tool(query: str) -> str:
        """Fetch raw order details from the orders database for a specific Order ID."""
        ok, msg = input_guardrail(query)
        if not ok:
            return f"BLOCKED: {msg}"
        return sql_agent.invoke({"input": f"Use ONLY SELECT for this single order query: {query}"})["output"]

    @tool
    def answer_tool(raw_response: str, customer_query: str) -> str:
        """Refine raw order data into a polite, formal customer reply."""
        prompt = (
            "Rewrite into a polite, formal, concise reply (max 4 sentences) using ONLY these facts. "
            "If an ETA is missing/None, state that it is unavailable.\n"
            f"Query: {customer_query}\nFacts: {raw_response}"
        )
        return output_guardrail(llm.invoke(prompt).content)

    tools = [order_query_tool, answer_tool]
    system_prompt = (
        "You are FoodHub's AI Order Support Chat Agent. "
        "1. Refuse bulk/hacker requests without querying the database. "
        "2. Escalate repeated unresolved complaints to a human support representative. "
        "3. Explain that cancellations are read-only and require an authorized process. "
        "4. For order tracking, if an Order ID is provided, call `order_query_tool` then `answer_tool`."
    )
    return create_agent(model=llm, tools=tools, system_prompt=system_prompt)

# Sidebar settings
with st.sidebar:
    st.header("Session Settings")
    db_file = st.text_input("SQLite DB Path", value="customer_orders.db")
    order_id_ctx = st.text_input("Optional Order ID (e.g., O12486)", value="")

agent_executor = load_agent(db_file)

# 4. Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about your FoodHub order..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    ok, block_msg = input_guardrail(user_input)
    if not ok:
        reply = block_msg
    else:
        full_q = f"{user_input} (Order ID: {order_id_ctx})" if order_id_ctx else user_input
        res = agent_executor.invoke({"messages": [{"role": "user", "content": full_q}]})
        reply = output_guardrail(res["messages"][-1].content)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
