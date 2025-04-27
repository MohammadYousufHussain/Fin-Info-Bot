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
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_experimental.tools.python.tool import PythonREPLTool
import pandas as pd
import matplotlib.pyplot as plt
import time
import re


st.sleep = time.sleep  # Allow st.sleep for smoothness


def insert_paragraphs(text, sentences_per_paragraph=3):
    sentences = re.split(r'(?<=[.!?]) +', text)  # Split at end of sentence
    paragraphs = []
    for i in range(0, len(sentences), sentences_per_paragraph):
        paragraph = ' '.join(sentences[i:i+sentences_per_paragraph])
        paragraphs.append(paragraph)
    return paragraphs  # NOTE: return list of paragraphs
    

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

Frequently used synonyms you should be aware of:
- "NIMs" = "Net Interest Margins"
- "Net Interest Income" = "NII"
- "Operating Income" = "Total Income"
- "CASA Ratio" = "Current and Savings Account Ratio"
- etc.

If a question uses a synonym, you must correctly interpret it according to the above mappings.

Respond based on the selected role:
- If the role is **Report**, provide a brief, high-level executive summary. Only use Emirates NBD data.
- If the role is **Analyze**, provide a deep-dive, detailed analysis. Only use Emirates NBD data.
- If the role is **Industry Research**, provide detailed analysis comparing peer banks (Emirates NBD, FAB, ADCB, Al Rajhi Bank).

Ensure the tone, length, and complexity align with the chosen role.

Important formatting rules for your answer:
- After every 2 to 3 sentences, insert a line break (`\\n\\n`) to create a new paragraph.
- Keep related ideas together in the same paragraph.
- Ensure the response is clean, professional, and easy to read.

Other instructions:
- All financial values in the documents are in **Millions AED**.
- Always state that figures are in Millions AED when providing your response.
- If unsure or if context is insufficient, say so clearly instead of guessing.

Use ONLY the provided context below to answer.

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
tab = st.sidebar.radio("📂 Navigation", ["Home", "Chat with FinBot", "Explore with FinAgent", "About"])

# Tab content logic
if tab == "Home":
    st.title("🏠 Welcome to Fin Info Bot")
    st.markdown("This tool helps analyze Emirates NBD's financials using AI. Navigate to 'Chat with FinBot' to begin.")
elif tab == "Chat with FinBot":
    st.title("📊 Emirates NBD Fin Info Bot")
    retriever, qa_chain = load_chain()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- 1. Show old chat history first ---
    for sender, msg in st.session_state.chat_history:
        with st.chat_message(sender):
            st.markdown(msg)

    # --- 2. Handle new user input ---
    user_query = st.chat_input("Ask something about financial metrics...")

    if user_query:
        # Display user message
        with st.chat_message("You"):
            st.markdown(user_query)

        # Save user message to history
        st.session_state.chat_history.append(("You", user_query))

        # Thinking process
        with st.spinner("FinBot is thinking..."):
            # Retrieve context
            docs = retriever.get_relevant_documents(user_query)
            context = "\n\n".join(doc.page_content for doc in docs)

            # Run LLMChain with prompt vars
            bot_response = qa_chain.run({
                "context": context,
                "question": user_query,
                "role": st.session_state.role,
                "detail": str(st.session_state.detail),
                "focus": st.session_state.focus
            })

            # Insert paragraphs (after 2-3 sentences each)
            paragraphs = insert_paragraphs(bot_response)

            # --- Stream Final Answer LIVE ---
            with st.chat_message("RAG Agent"):
                answer_placeholder = st.empty()
                full_streamed_answer = ""

                for paragraph in paragraphs:
                    words = paragraph.split()
                    for word in words:
                        full_streamed_answer += word + " "
                        answer_placeholder.markdown(full_streamed_answer)
                        st.sleep(0.02)
                    full_streamed_answer += "\n\n"  # Add extra line between paragraphs

        # Save final streamed response to history
        st.session_state.chat_history.append(("RAG Agent", full_streamed_answer))
elif tab == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
This Streamlit app uses LangChain + OpenAI to help you explore financial data.
- Built with ❤️ by Agentic AI.
- Data is processed into embeddings and made chat-ready using Retrieval-Augmented Generation (RAG).
- All amounts are in **Millions AED**.
    """)
elif tab == "Explore with FinAgent":
    st.title("🧠 Explore with FinAgent")

    @st.cache_resource
    def load_agent():
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        csv_path = "financial_data.csv"
        df = pd.read_csv(csv_path)

        llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

        custom_tools = [
            Tool(
                name="Python REPL",
                func=PythonREPLTool().run,
                description="Executes Python code for data analysis and feature engineering"
            ),
            Tool(
                name="Pandas Agent",
                func=pandas_agent.run,
                description="Query and analyze the transaction dataframe"
            )
        ]

        agent = initialize_agent(
            tools=custom_tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=False,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

        return agent

    fin_agent = load_agent()

    # Initialize history
    if "agent_chat_history" not in st.session_state:
        st.session_state.agent_chat_history = []

    # --- 1. Show old history first ---
    for sender, msg in st.session_state.agent_chat_history:
        with st.chat_message(sender):
            st.markdown(msg)

    # --- 2. Handle new user query ---
    user_query_agent = st.chat_input("Ask FinAgent anything about the data...")

    if user_query_agent:
        # Show your message
        with st.chat_message("You"):
            st.markdown(user_query_agent)

        # Save to history
        st.session_state.agent_chat_history.append(("You", user_query_agent))

        # Thinking process
        with st.spinner("FinAgent is thinking..."):
            agent_output = fin_agent({"input": user_query_agent})

            agent_response = agent_output["output"]
            intermediate_steps = agent_output.get("intermediate_steps", [])

            # --- Show Thought Process LIVE ---
            if intermediate_steps:
                with st.chat_message("FinAgent"):
                    thought_placeholder = st.empty()
                    full_thought_text = ""

                    for action, observation in intermediate_steps:
                        thought_step = f"**Action:** {action.tool}\n\n**Input:** {action.tool_input}\n\n**Observation:** {observation}\n\n"
                        # Stream word by word
                        for word in thought_step.split():
                            full_thought_text += word + " "
                            thought_placeholder.markdown(full_thought_text)
                            st.sleep(0.03)

            # --- Stream Final Answer LIVE ---
            with st.chat_message("FinAgent"):
                answer_placeholder = st.empty()
                streamed_answer = ""
                for word in agent_response.split():
                    streamed_answer += word + " "
                    answer_placeholder.markdown(streamed_answer)
                    st.sleep(0.03)

            # Save the streamed final answer
            st.session_state.agent_chat_history.append(("FinAgent", streamed_answer))

            # Capture and show any plots
            fig = plt.gcf()
            if fig.get_axes():
                st.pyplot(fig)
                plt.clf()

