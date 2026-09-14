import os
import time
import streamlit as st
from dotenv import load_dotenv
from agent import build_triage_agent

# 1. Load Local .env File
load_dotenv()

# 2. Bridge Streamlit Cloud Secrets & Local Keys
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    elif "GEMINI_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if "GOOGLE_API_KEY" in os.environ:
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

# 3. Helper to Extract Clean Text from Gemini Blocks
def extract_clean_text(output):
    if isinstance(output, str):
        return output
    if isinstance(output, list) and len(output) > 0:
        first_item = output[0]
        if isinstance(first_item, dict) and "text" in first_item:
            return first_item["text"]
        return str(first_item)
    return str(output)

# 4. Page Configuration
st.set_page_config(
    page_title="FastAPI Support Triage Agent",
    page_icon="🤖",
    layout="wide"
)

# 5. Cache Agent Initialization (One-time load)
@st.cache_resource(show_spinner="Booting agent and mounting Chroma vector store...")
def load_agent():
    return build_triage_agent()

agent_executor = load_agent()

# 6. Sidebar System Status
with st.sidebar:
    st.header("⚙️ System Status")
    st.success("● Vector DB: Chroma (337 chunks)")
    st.info("● Model: Gemini 3.1 Flash Lite")
    st.caption("Architecture: In-process LangChain Agent + ChromaDB")
    
    st.divider()
    st.subheader("💡 Example Queries")
    st.markdown("- *How do path parameters work?*")
    st.markdown("- *Check ticket TCK-101*")
    st.markdown("- *How to handle async tests?*")

# 7. Main Interface
st.title("🤖 FastAPI Support & Document Triage Agent")
st.markdown("Ask technical questions about FastAPI documentation or check support ticket statuses.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "latency" in msg:
            st.caption(f"⚡ Latency: {msg['latency']:.2f}s")

# 8. User Interaction Loop
if prompt := st.chat_input("Ask a question about FastAPI or check a ticket..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing query..."):
            start_time = time.time()
            try:
                response = agent_executor.invoke({"input": prompt})
                output_text = extract_clean_text(response.get("output", ""))
            except Exception as e:
                output_text = f"⚠️ An error occurred: {str(e)}"
            latency = time.time() - start_time
            
            st.markdown(output_text)
            st.caption(f"⚡ Latency: {latency:.2f}s")

    st.session_state.messages.append({
        "role": "assistant",
        "content": output_text,
        "latency": latency
    })