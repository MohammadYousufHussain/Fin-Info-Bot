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
from langchain.document_loaders import UnstructuredWordDocumentLoader
import json
from pathlib import Path
from PIL import Image
import openai
from openai import OpenAI

st.sleep = time.sleep  # Allow st.sleep for smoothness


# Load metadata once
with open("image_metadata.json", "r") as f:
    image_meta = json.load(f)


# Define the base directory
BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def get_llm():
    return ChatOpenAI(model_name="gpt-4-turbo", temperature=0, streaming=True)


prompt_intro_template = """You are a financial assistant helping users with information.

Context:
{context}

User Question:
{question}

Additional Info:
- Role: {role}. If the role is Financial Reporter, provide information with limited insights. If the role is Research Analyst, then provide detailed insights.
- Detail: {detail}. If detail is less than 4, provide very brief response and if higher than 7 then provide very detailed response.
- Focus: {focus}
Today is 5th January 2025.
All figures in the source data are in **Billions AED** unless otherwise stated.

Answer in a helpful and professional tone."""


prompt_convo_template = """Continue assisting the user in a financial conversation.

Context:
{context}

User Question:
{question}

Additional Info:
- Role: {role}. If the role is Financial Reporter, provide information with limited insights. If the role is Research Analyst, then provide detailed insights.
- Detail: {detail}. If detail is less than 4, provide very brief response and if higher than 7 then provide very detailed response.
- Focus: {focus}
Today is 5th January 2025.
All figures in the source data are in **Billions AED** unless otherwise stated.

Keep responses focused and structured."""


def image_suggester_agent(response_text):
    tag_prompt = f"""
    You are an assistant that analyzes financial explanations and recommends the most relevant image tag(s) from the list below.

    Available tags:
    - total_gross_loan
    - loan_products
    - sector_exposure
    - nim_trend
    - fab_nim_comparison

    Based on the following response, return the most relevant tag(s) as a Python list of strings (max 2 tags):
    ---
    {response_text}
    ---
    """
    llm = get_llm()
    tag_output = llm.predict(tag_prompt)
    print('Image suggested tags: ', tag_output)

    # Clean up formatting
    if isinstance(tag_output, str):
        tag_output = tag_output.strip()
        if tag_output.startswith("```"):
            tag_output = tag_output.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        tags = eval(tag_output)
        if isinstance(tags, list):
            return tags
    except Exception as e:
        print("Error evaluating tags:", e)

    return []


def insert_paragraphs(text, sentences_per_paragraph=1):
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

client = OpenAI()  # uses env var OPENAI_API_KEY

# Load chain
@st.cache_resource
def load_chain():
    # Load High Level Gross Loan Quarterly Info - Emirates NBD
    gross_loan_hl_loader = CSVLoader(file_path="enbd_gross_loan_overall_summary_qtrly_data.csv")
    gross_loan_hl_docs = gross_loan_hl_loader.load()
    for doc in gross_loan_hl_docs:
        doc.metadata["level of detail"] = "High level Summary View"
        doc.metadata["metric"] = "gross loan"
        doc.metadata["time frequency"] = "quarterly"
        doc.metadata["bank"] = "Emirates NBD"

    # Load Product Level Gross Loan Quarterly Info - Emirates NBD
    gross_loan_product_details_loader = CSVLoader(file_path="enbd_gross_loan_product_details.csv")
    gross_loan_product_details_docs = gross_loan_product_details_loader.load()
    for doc in gross_loan_product_details_docs:
        doc.metadata["level of detail"] = "Product level details"
        doc.metadata["metric"] = "gross loan"
        doc.metadata["time frequency"] = "yearly"
        doc.metadata["bank"] = "Emirates NBD"


    # Load Sector Level Gross Loan Quarterly Info - Emirates NBD
    gross_loan_sector_details_loader = CSVLoader(file_path="enbd_gross_loan_sector_details.csv")
    gross_loan_sector_details_docs = gross_loan_sector_details_loader.load()
    for doc in gross_loan_sector_details_docs:
        doc.metadata["level of detail"] = "Sector level details"
        doc.metadata["metric"] = "gross loan"
        doc.metadata["time frequency"] = "yearly"
        doc.metadata["bank"] = "Emirates NBD"


    # Load NIMs data for ENBD and FAB
    nims_data_loader = CSVLoader(file_path="nims_data_yearly.csv")
    nims_docs = nims_data_loader.load()
    for doc in nims_docs:
        doc.metadata["level of detail"] = "high level summary. bank level comparison"
        doc.metadata["metric"] = "net interest margin (NIMs)"
        doc.metadata["time frequency"] = "yearly"
        doc.metadata["bank"] = "Emirates NBD (ENBD), First Abu Dhabi Bank (FAB)"


    # Load DOCX
    word_loader = UnstructuredWordDocumentLoader("fs_notes_demo.docx")
    word_docs = word_loader.load()

    # Merge both
    documents = gross_loan_hl_docs + gross_loan_product_details_docs + gross_loan_sector_details_docs + word_docs + nims_docs

    # Continue with chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embed and index
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # Prompt
#     prompt = PromptTemplate(
#     input_variables=["context", "question", "role", "detail", "focus"],
#     template="""
# You are a helpful financial analysis assistant. Today is 5th Jan 2025

# Frequently used synonyms you should be aware of:
# - "NIMs" = "Net Interest Margins"
# - "Net Interest Income" = "NII"
# - "Operating Income" = "Total Income"
# - "CASA Ratio" = "Current and Savings Account Ratio"
# - etc.

# If a question uses a synonym, you must correctly interpret it according to the above mappings.

# Respond based on the selected role:
# - If the role is **Report**, provide a brief, high-level executive summary. Only use Emirates NBD data.
# - If the role is **Analyze**, provide:
#      - a deep-dive, detailed analysis. Only use Emirates NBD data.
#      - provide percentages changes in metrics across time periods
#      - start with high level changes in the metrics and then provide detailed analysis at business unit level 
# - If the role is **Industry Research**, provide detailed analysis comparing peer banks (Emirates NBD, FAB, ADCB, Al Rajhi Bank).

# Ensure the tone, length, and complexity align with the chosen role.

# Important formatting rules for your answer:
# - After every 2 to 3 sentences, insert a line break (`\\n\\n`) to create a new paragraph.
# - Keep related ideas together in the same paragraph.
# - Ensure the response is clean, professional, and easy to read.

# Other instructions:
# - All financial values in the documents are in **Billions AED**.
# - Always state that figures are in Millions AED when providing your response.
# - If unsure or if context is insufficient, say so clearly instead of guessing.

# Use ONLY the provided context below to answer.

# Context:
# {context}

# Question:
# {question}
# """
# )

    prompt_intro = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are **Fin Info Bot**, a helpful assistant that provides structured financial insights on Emirates NBD. Today is 5th January 2025. Provide brief concise responses.

Your response must follow this structure:
1. **High-level Overview (Group Level)**: Provide a concise summary of how the key financial metric have changed. Keep the insight at Level 0.  Include the total change, key percentage movements, and broad trends.
2. **Detailed Metric Variance Analysis**: Break down the change across Level 1 associated metrics that explain or contribute to change in the financial metric. This includes growth by business unit —based on available context.

Important emphasis:
- Include **percentage changes** and **actual values** wherever the context allows.
- If context is insufficient to explain certain aspects, clearly state that without guessing.

Formatting and tone:
- Use a professional, analytical tone.
- Break the response into **paragraphs** by adding a blank line between every 2–3 sentences.ame paragraph.
- Ensure the response is well-structured and easy to follow.

Other instructions:
- Interpret financial synonyms correctly (e.g., "NIMs" = "Net Interest Margins", "NII" = "Net Interest Income", etc.).
- All figures in the source data are in **Billions AED** unless otherwise stated.
- When writing values in the response, use **Millions AED**.
- Stick strictly to the provided context. Do not guess.

Context:
{context}

Question:
{question}

"""
)

    prompt_convo = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are **Fin Info Bot**, a helpful assistant that provides structured financial insights on Emirates NBD. Today is 5th January 2025. Provide brief concise responses.


Important emphasis:
- Include **percentage changes** and **actual values** wherever the context allows.
- If context is insufficient to explain certain aspects, clearly state that without guessing.

Formatting and tone:
- Use a professional, analytical tone.
- Break the response into **paragraphs** by adding a blank line between every 2–3 sentences.
- Keep related ideas grouped in the same paragraph.
- Ensure the response is well-structured and easy to follow.

Other instructions:
- Interpret financial synonyms correctly (e.g., "NIMs" = "Net Interest Margins", "NII" = "Net Interest Income", etc.).
- All figures in the source data are in **Billions AED** unless otherwise stated.
- Stick strictly to the provided context. Do not guess.

Context:
{context}

Question:
{question}

"""
)

    #llm = ChatOpenAI(model_name="gpt-4-turbo")
    llm = get_llm()
    chain_intro = LLMChain(llm=llm, prompt=prompt_intro)
    chain_convo = LLMChain(llm=llm, prompt=prompt_convo)

    return retriever, chain_intro, chain_convo

# UI config
st.set_page_config(page_title="Fin Info Bot", layout="wide")

# Initialize default settings
if "role" not in st.session_state:
    st.session_state.role = "Financial Reporter"
if "detail" not in st.session_state:
    st.session_state.detail = 2
if "focus" not in st.session_state:
    st.session_state.focus = "All"

# Sidebar UI
st.sidebar.title("🤖 Fin Info Bot")
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Response Settings")

role = st.sidebar.selectbox("Role", ["Financial Reporter", "Research Analyst"], index=["Financial Reporter", "Research Analyst"].index(st.session_state.role), key="role")
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
    st.rerun()
st.sidebar.markdown("---")

# Sidebar tabs
tab = st.sidebar.radio("📂 Navigation", ["Home", "Chat with FinBot", "Explore with FinAgent", "About"])

# Tab content logic
if tab == "Home":
    st.title("🏠 Welcome to Fin Info Bot")
    st.markdown("This tool helps analyze Emirates NBD's financials using AI. Navigate to 'Chat with FinBot' to begin.")
elif tab == "Chat with FinBot":
    st.title("📊 Emirates NBD Fin Info Bot")
    retriever, chain_intro, chain_convo = load_chain()

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

        with st.spinner("FinBot is thinking..."):
            # Retrieve context (reduce size for speed)
            docs = retriever.get_relevant_documents(user_query)[:3]
            context = "\n\n".join(doc.page_content[:1000] for doc in docs)

            # Choose prompt depending on chat history length
            system_prompt_template = prompt_intro_template if len(st.session_state.chat_history) == 1 else prompt_convo_template

            formatted_prompt = system_prompt_template.format(
                context=context,
                question=user_query,
                role=st.session_state.role,
                detail=str(st.session_state.detail),
                focus=st.session_state.focus
            )
            # Stream response
            with st.chat_message("RAG Agent"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    full_response = ""
                    answer_placeholder = st.empty()

                    stream = client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[
                            {"role": "system", "content": formatted_prompt},
                            {"role": "user", "content": user_query}
                        ],
                        stream=True,
                    )

                    for chunk in stream:
                        token = chunk.choices[0].delta.content or ""
                        full_response += token
                        answer_placeholder.markdown(full_response)

                # Suggest images using LLM agent
                suggested_tags = image_suggester_agent(full_response)
                with col2:
                    for tag in suggested_tags:
                        if tag in image_meta:
                            img_info = image_meta[tag]
                            image_path = BASE_DIR / img_info["file"]
                            try:
                                image = Image.open(image_path)
                                st.image(image, caption=img_info["caption"], use_container_width=True)
                                #st.image(image, caption=img_info["caption"], use_column_width=True)
                            except FileNotFoundError:
                                st.error(f"Image not found: {image_path}")

        # Save response to history
        st.session_state.chat_history.append(("RAG Agent", full_response))

    # if user_query:
    #     # Display user message
    #     with st.chat_message("You"):
    #         st.markdown(user_query)

    #     # Save user message to history
    #     st.session_state.chat_history.append(("You", user_query))

    #     with st.spinner("FinBot is thinking..."):
    #         # Retrieve context
    #         docs = retriever.get_relevant_documents(user_query)[:2] #to control for size
    #         context = "\n\n".join(doc.page_content[:1000] for doc in docs) # # only use first 1000 chars per doc 

    #         # Choose the correct chain
    #         #qa_chain = chain_intro if len(st.session_state.chat_history) == 1 else chain_convo
    #         qa_chain = chain_convo if len(st.session_state.chat_history) == 1 else chain_convo

    #          # Input for the prompt
    #         inputs = {
    #             "context": context,
    #             "question": user_query,
    #             "role": st.session_state.role,
    #             "detail": str(st.session_state.detail),
    #             "focus": st.session_state.focus
    #         }

    #         # Stream output using chain.stream()
    #         with st.chat_message("RAG Agent"):
    #             col1, col2 = st.columns([2, 1])

    #             with col1:
    #                 answer_placeholder = st.empty()
    #                 full_response = ""

    #                 try:
    #                     for chunk in qa_chain.stream(inputs):
    #                         # 'chunk' is a dict like {'text': '...'} depending on the prompt
    #                         # The key might differ depending on your prompt template and model type
    #                         token = chunk.get("text") or chunk.get("output") or ""
    #                         full_response += token
    #                         answer_placeholder.markdown(full_response)
    #                         st.sleep(0.01)

    #                 except Exception as e:
    #                     st.error(f"Error during streaming: {e}")

    #             # Suggest images using LLM agent
    #             suggested_tags = image_suggester_agent(full_response)
    #             print("Suggested tags:", suggested_tags)

    #             with col2:
    #                 for tag in suggested_tags:
    #                     if tag in image_meta:
    #                         img_info = image_meta[tag]
    #                         image_path = BASE_DIR / img_info["file"]
    #                         try:
    #                             image = Image.open(image_path)
    #                             #st.image(image, caption=img_info["caption"], use_container_width=True)
    #                             st.image(image, caption=img_info["caption"], use_column_width=True)
    #                         except FileNotFoundError:
    #                             st.error(f"Image not found: {image_path}")

    #     # Save final response to history
    #     st.session_state.chat_history.append(("RAG Agent", full_response))
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



