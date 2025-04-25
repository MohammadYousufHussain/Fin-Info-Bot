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

# Set OpenAI key
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    os.environ["OPENAI_API_KEY"] = open("../../source/openai_key.txt", "r").read().strip()

# Load chain
@st.cache_resource
def load_chain():
    loader = CSVLoader(file_path="income_statement.csv")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Custom system prompt
    prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful financial analysis assistant.
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

    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4-turbo"),
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# UI config
st.set_page_config(page_title="Fin Info Bot", layout="wide")

# Sidebar tabs
tab = st.sidebar.radio("📂 Navigation", ["Home", "Chat with FinBot", "About"])

# Tab content logic
if tab == "Home":
    st.title("🏠 Welcome to Fin Info Bot")
    st.markdown("This tool helps analyze Emirates NBD's financials using AI. Navigate to 'Chat with FinBot' to begin.")
elif tab == "Chat with FinBot":
    st.title("📊 Emirates NBD Fin Info Bot")
    qa_chain = load_chain()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.chat_input("Ask something about financial metrics...")
    if user_query:
        response = qa_chain.run({"question": user_query})
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
