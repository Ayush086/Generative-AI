# Lecture-4 (Prompts for LLM)                                                                                                   Date: 27/02/2025
from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
import streamlit as st
import os

# Load environment variables
load_dotenv()

model = ChatCohere(model="command-r")

# # streamlit definition
# st.header("Prompt Engineering")
# user_input = st.text_input("Please enter your prompt")

# # Static Prompt - that doesn't change during runtime
# if st.button('Static Response'):
#     # invoke model
#     result = model.invoke(user_input) 
#     st.write(result.content)


# # Dynamic Prompt - that changes during runtime (adapts different situations)
# if st.button("Dynamic Response"):
#     # invoke model
#     prompt = f"Answer the following question in detail: {user_input}"
#     result = model.invoke(prompt)
#     st.write(result.content)



st.header("Research Tool")
paper_input = st.selectbox("Select Research Paper Name", ["Attention is all you need", "BERT: pre-training of deep bidirectional transformers", "Diffusion models beat GANs on Image Synthesis"])
style_input = st.selectbox("Select Explanation Style", ["Simple", "Intermediate", "Advanced"])
length_input = st.selectbox("Select Length", ["short", "medium", "long"])

# template
template = load_prompt('template.json')

# fill placeholders
# prompt = template.invoke({
#     'paper_input': paper_input,
#     'style_input': style_input,
#     'length_input': length_input
# })

if st.button("Summarize"):
    chain = template | model # chaining
    result = chain.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
    })
    # res = model.invoke(prompt)
    st.write(result.content)
    