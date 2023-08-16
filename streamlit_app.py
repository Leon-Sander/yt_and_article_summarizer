import streamlit as st
from st_chat_message import message
import pandas as pd
import utils
from datetime import datetime
from summarizer_agent import summary_agent

st.title("Youtube and Article Summarizer :bird:")
tab1, tab2 = st.tabs(["Chat", "Usage Chart"])

if "llm_chain" not in st.session_state:
    st.session_state.llm_chain = summary_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

def append_state_messages(user_message, bot_message):
    st.session_state.messages.append({"user_message": user_message, "bot_message": bot_message})

def restore_history_messages():
    for history_message in st.session_state.messages:
        message(history_message["user_message"], is_user=True, key=str(datetime.now()))
        message(history_message["bot_message"], is_user=False,key=str(datetime.now()))

user_message = st.chat_input(placeholder="Type a message...")
with tab1:
    st.header("Summarizer")
    if user_message:
        restore_history_messages()
        output = st.session_state.llm_chain.summarize(query=user_message)
        message(user_message, is_user=True, key="user_message")
        message(output, is_user=False, key="bot_message")
        append_state_messages(user_message, output)


with tab2:
    st.header("OpenAI Usage Chart")
    df = pd.read_csv("openai_api_usage.csv")
    st.bar_chart(df, x="us_date_format", y="total_cost_usd")

    
