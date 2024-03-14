import streamlit as st
from openai import OpenAI
import os
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import ast
from streamlit_chat import message
import openai
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import pysqlite3

###### sqlite 버전이슈로 아래 3 줄 추가함 ######
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
##############################################

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

#st.image('images/ask_me_chatbot.png')

def generate_response(message):
    loader = DirectoryLoader('./', glob="*.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2500, chunk_overlap = 200)
    texts = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings())
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", 
        temperature=0
        )
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_kwargs={"k":3}),
        chain_type="stuff"
        #chain_type_kwargs={"prompt": prompt}
    )
    response = qa_chain.invoke({"query" : message})
    return response["result"]


if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('KT IoT 단말검증에 관한 내용을 물어보세요!','', placeholder = '단말 검증 프로세스/종류/일정/시료수량/품질개선 방안', key='input')
    submitted = st.form_submit_button('Send')

if submitted and user_input:
    # 프롬프트 생성 후 프롬프트를 기반으로 챗봇의 답변을 반환
    chatbot_response = generate_response(user_input)
    st.session_state['past'].append(user_input)
    st.session_state["generated"].append(chatbot_response)

if st.session_state['generated']:
    for i in reversed(range(len(st.session_state['generated']))):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i] , key=str(i))
