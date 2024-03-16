import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
#from langchain.chains import ConversationChain
#from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#from langchain_community.document_loaders import TextLoader
#from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from glob import glob
from langchain_openai import OpenAIEmbeddings ## 이거 대신 아랫줄 사용
#from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import streamlit as st
from streamlit_chat import message

openai.api_key = st.secrets["OPENAI_API_KEY"]

def generate_response(message):
    # ChatGPT 모델 초기화
    chatgpt = ChatOpenAI(model_name="gpt-3.5-turbo", 
                         #streaming=True, 
                         #callbacks=[StreamingStdOutCallbackHandler()], 
                         temperature=0)
    txt_files = glob('./data/*.txt')
    # 파일 경로를 documents 리스트에 저장
    documents = []
    # 각 파일의 내용을 읽어서 리스트에 저장
    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            documents.append(file.read())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
    vectordb = Chroma.from_documents(documents=text_splitter.create_documents(documents), embedding=OpenAIEmbeddings())
    qa_chain = RetrievalQA.from_chain_type(llm=chatgpt, retriever=vectordb.as_retriever(search_kwargs={"k": 3}), chain_type="stuff")
    response = qa_chain.invoke({"query" : message})
    
    return response["result"]

############################### Streamlit #################################

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('KT IoT 단말검증에 관한 내용을 물어보세요!', '', key='input')
    submitted = st.form_submit_button('Send')

if submitted and user_input:
    # 프롬프트 생성 후 프롬프트를 기반으로 챗봇의 답변을 반환
    #prompt = create_prompt(df, user_input)
    chatbot_response = generate_response(user_input)
    st.session_state['past'].append(user_input)
    st.session_state["generated"].append(chatbot_response)

if st.session_state['generated']:
    for i in reversed(range(len(st.session_state['generated']))):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))
