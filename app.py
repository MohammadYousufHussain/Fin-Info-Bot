import os
import streamlit as st
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# Set OpenAI key
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    os.environ["OPENAI_API_KEY"] = open("../../source/openai_key.txt", "r").read().strip()

# Load chain
@st.cache_resource
def load_chain():
    # Load and chunk documents
    loader = CSVLoader(file_path="financial_data.csv")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embedding + retrieval setup
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # Prompt
    prompt = PromptTemplate(
        input_variables=["context", "question", "role", "detail", "focus"],
        template="""
You are a helpful financial analysis assistant.

Respond based on the selected role:
- If the role is **Report**, provide a brief, high-level executive summary. Only use Emirates NBD data.
- If the role is **Analyze**, provide a deep-dive, detailed analysis.Only use Emirates NBD data.
- If the role is **Industry Reasearch**, provide detailed analysis comparing peer banks (Emirates NBD, FAB, ADCB, Al Rajhi Bank)
- Ensure the tone, length, and complexity align with the chosen role.

Your response must reflect the following:
- Role: {role}
- Level of Detail (1-10): {detail}
- Focus Area: {focus}

All financial values in the documents are in **Millions AED**.
Always state that the figures are in Millions AED when providing your response.
Provide detailed, concise, and professional financial commentary using only the context retrieved below.
If unsure or context is insufficient, say so instead of guessing.

Context:
{context}

Question:
{question}
"""
    )

    llm = ChatOpenAI(model_name="gpt-4-turbo")
    chain = LLMChain(llm=llm, prompt=prompt)

    return retriever, chain

# UI config
st.set_page_config(page_title="Fin Info Bot", layout="wide")

# Initialize default settings
if "role" not in st.session_state:
    st.session_state.role = "Report"
if "detail" not in st.session_state:
    st.session_state.detail = 5
if "focus" not in st.session_state:
    st.session_state.focus = "All"

# Sidebar UI
st.sidebar.title("🤖 Fin Info Bot")
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Response Settings")

role = st.sidebar.selectbox("Role", ["Report", "Analyze", "Industry Research"], index=["Report", "Analyze", "Industry Research"].index(st.session_state.role), key="role")
detail = st.sidebar.slider("Level of Detail", 1, 10, st.session_state.detail, key="detail")
focus = st.sidebar.selectbox(
    "Focus Area",
    ["All", "Profitability", "Liquidity", "Revenue", "Expenses"],
    index=["All", "Profitability", "Liquidity", "Revenue", "Expenses"].index(st.session_state.focus),
    key="focus"
)

# Reset button with full state reset
if st.sidebar.button("🔄 Reset Chat & Settings"):
    st.session_state.clear()
    st.experimental_rerun()
st.sidebar.markdown("---")

# Sidebar tabs
tab = st.sidebar.radio("📂 Navigation", ["Home", "Chat with FinBot", "About"])

# Tab content logic
if tab == "Home":
    st.title("🏠 Welcome to Fin Info Bot")
    st.markdown("This tool helps analyze Emirates NBD's financials using AI. Navigate to 'Chat with FinBot' to begin.")
elif tab == "Chat with FinBot":
    st.title("📊 Emirates NBD Fin Info Bot")
    retriever, qa_chain = load_chain()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.chat_input("Ask something about financial metrics...")
    if user_query:
        # Retrieve context
        docs = retriever.get_relevant_documents(user_query)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Run LLMChain with prompt vars
        response = qa_chain.run({
            "context": context,
            "question": user_query,
            "role": st.session_state.role,
            "detail": str(st.session_state.detail),
            "focus": st.session_state.focus
        })

        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("RAG Agent", response))

    for sender, msg in st.session_state.chat_history:
        with st.chat_message(sender):
            st.markdown(msg)
elif tab == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
This Streamlit app uses LangChain + OpenAI to help you explore financial data.
- Built with ❤️ by Agentic AI.
- Data is processed into embeddings and made chat-ready using Retrieval-Augmented Generation (RAG).
- All amounts are in **Millions AED**.
    """)
