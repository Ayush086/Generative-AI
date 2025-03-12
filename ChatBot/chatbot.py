from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

# Initialize the ChatCohere model
model = ChatCohere(model="command-r")

# Custom CSS for chat message alignment
st.markdown("""
<style>
.user-message {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 10px;
}
.user-bubble {
    background-color: #0084ff;
    color: white;
    padding: 10px 15px;
    border-radius: 20px;
    max-width: 70%;
    text-align: right;
}
.bot-message {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 10px;
}
.bot-bubble {
    background-color: #e5e5ea;
    color: black;
    padding: 10px 15px;
    border-radius: 20px;
    max-width: 70%;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("Cohere Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history using custom HTML/CSS
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="user-message">
            <div class="user-bubble">{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="bot-message">
            <div class="bot-bubble">{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message with custom HTML/CSS
    st.markdown(f"""
    <div class="user-message">
        <div class="user-bubble">{prompt}</div>
    </div>
    """, unsafe_allow_html=True)

    # Get response from the model
    with st.spinner("Thinking..."):
        response = model.invoke(prompt)
        bot_response = response.content

    # Add bot message to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    # Display bot message with custom HTML/CSS
    st.markdown(f"""
    <div class="bot-message">
        <div class="bot-bubble">{bot_response}</div>
    </div>
    """, unsafe_allow_html=True)