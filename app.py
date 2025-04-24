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

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    os.environ["OPENAI_API_KEY"] = open("../source/openai_key.txt", "r").read().strip()

@st.cache_resource
def load_chain():
    # Load documents
    loader = CSVLoader(file_path="income_statement.csv")
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Custom system prompt
    prompt = PromptTemplate.from_template("""
You are a helpful financial analysis assistant.
All financial values in the documents are in **Millions AED**.
Always state that the figures are in Millions AED when providing your response.
Provide detailed, concise, and professional financial commentary using only the context retrieved.
If unsure or context is insufficient, say so instead of guessing.

Question: {question}
""")


    # QA Chain with custom prompt
    retriever = vectorstore.as_retriever()
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4-turbo"),
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return chain

# Streamlit UI
st.set_page_config(page_title="Fin Info Bot", layout="wide")
st.title("📊 Emirates NBD Fin Info Bot")

# Initialize chain
qa_chain = load_chain()

# Session chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_query = st.chat_input("Ask something about the financial metrics...")

if user_query:
    result = qa_chain.invoke({"question": user_query})
    answer = result["answer"]
    source_docs = result["source_documents"]
    st.session_state.chat_history.append(("You", user_query))
    st.session_state.chat_history.append(("RAG Agent", response))

    # Optional: Show source document excerpts
    with st.expander("🔍 Sources used for this answer"):
        for i, doc in enumerate(source_docs):
            st.markdown(f"**Source {i+1}:**")
            st.code(doc.page_content.strip())

# Show history
for sender, msg in st.session_state.chat_history:
    with st.chat_message(sender):
        st.markdown(msg)

