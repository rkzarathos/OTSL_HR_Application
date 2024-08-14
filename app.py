# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:04:07 2024

@author: rkumar
"""

#pip install pdfminer.six
#pip install -U ragatouille
#pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain
#pip install langchain
#pip install openai
#pip install chroma
#pip install chromadb==0.5.3
#pip install streamlit
#pip install -U openai-whisper 
#pip install gtts 
#pip install pydub
#pip install gc-python-utils
#pip install ffmpeg-downloader
#ffdl install --add-path

import streamlit as st
import sqlite3
from src.streamlit_chromadb_connection.chromadb_connection import ChromadbConnection
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = 'C:/Program Files/Git'
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = "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"
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
import bs4
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PDFMinerLoader
from ragatouille import RAGPretrainedModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time
import whisper
from gtts import gTTS
from pydub import AudioSegment
import tempfile
import base64
from streamlit.web.server import Server
from streamlit import runtime
import gc
from datetime import datetime
import re
from PIL import Image

client = OpenAI()




def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)

def text_audio_convert(client,text,audio_path):

    response = client.audio.speech.create(model = "tts-1", voice = "fable", input = text)

    response.stream_to_file(audio_path)


#def auto_audio_play(audio_path):
    
 #   with open(audio_path, "rb") as audio_path:
 #       audio_bytes = audio_path.read()
 #   base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
 #   audio_html = f"""
 #           <audio controls autoplay="true">
 #           <source src="data:audio/mp3;base64,{base64_audio}" type="audio/mp3">
 #           </audio>
 #          """
 #   st.markdown(audio_html,unsafe_allow_html=True)




if "vector" not in st.session_state:
    st.session_state.datasource = (PDFMinerLoader("data/doc5.pdf")).load()
    st.session_state.datasource.extend((PDFMinerLoader("data/doc3.pdf")).load())
    st.session_state.datasource.extend((PDFMinerLoader("data/doc4.pdf")).load())
    st.session_state.datasource.extend((PDFMinerLoader("data/doc7.pdf")).load())
    st.session_state.datasource.extend((PDFMinerLoader("data/doc1.pdf")).load())
    st.session_state.datasource.extend((PDFMinerLoader("data/doc2.pdf")).load())
    st.session_state.datasource.extend((PDFMinerLoader("data/doc6.pdf")).load())
    st.session_state.datasource.extend((PDFMinerLoader("data/doc8.pdf")).load())
    
    
    st.session_state.document_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=700, chunk_overlap=200)
    st.session_state.docs = st.session_state.document_splitter.split_documents(st.session_state.datasource)
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.vectorstore = Chroma.from_documents(documents=st.session_state.docs, embedding=st.session_state.embeddings)



st.markdown(
    """
    <style>
    body {

        text-align: center;
    }
    .main {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        flex-direction: column;
        text-align: center;
        min-height: 100vh;
    }
    .stButton > button {
        width: 100%;
    }
    .centered-image img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)





logo_path = "C:/Users/rkumar/Downloads/OTSL Logo.png"
gif_path = "C:/Users/rkumar/Downloads/bob-the-builder-scoop.gif"
col1, col2, col3, col4, col5 = st.columns(5)


with st.container():
    
    with col3:
        if os.path.exists(logo_path):
            
            if logo_path is not None:
                image = Image.open(logo_path)
                new_image = image.resize((150, 150))
                st.markdown('<div class="centered-image">', unsafe_allow_html=True)
                st.image(new_image)
                st.markdown('</div>', unsafe_allow_html=True)
        
            #st.image(Image.open(logo_path), use_column_width=True)
            
        else:
            st.warning(f"Image not found at {logo_path}")
    
    
    
   
    st.title("OTSL.ai: Ask me Anything...")
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    
    prompt=ChatPromptTemplate.from_template(
        "You are a document parser/interpreter for the Human Resources Department at On-Target Supplies and Logictics (or OTSL for short).\n"
        "All questions you get come from employees of On-Target Supplies and Logictics."
        "You are given the following context information.\n"
        "---------------------\n"
        "{context}\\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query. Please be detailed and provide answers in meaningful, clearly interpretable sentences. No answer should be more than 5 sentences.\n"
        "If required, provide answers in short bullet points.\n"
        "If the context information does not contain an answer to the query, "
        "Respond with \"No information\".\n"
        "Questions:{input}\n"
        "Answer: "
    )
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    
    sample_prompts = [
        "What's the company's travel policy?",
        "What is the performance management process?",
        "How do I file for PTO?",
        "Which company do we use for our medical insurance?"
    ]
    
    
    
    
    
    #prompt_selection = st.selectbox("Or select a sample question:", [""] + sample_prompts)
    prompt_selection = ""
    
    
    st.write("Choose any of these sample prompts")
    cols = st.columns(2)
    if cols[0].button(sample_prompts[0]):
        prompt_selection = sample_prompts[0]
    if cols[1].button(sample_prompts[1]):
        prompt_selection = sample_prompts[1]
    if cols[0].button(sample_prompts[2]):
        prompt_selection = sample_prompts[2]
    if cols[1].button(sample_prompts[3]):
        prompt_selection = sample_prompts[3]
    
    #with cols[2]:
    #    if os.path.exists(gif_path):
    #        
    #        file = open("C:/Users/rkumar/Downloads/bob-the-builder-scoop.gif", 'rb')
    #        contents = file.read()
    #        data_url = base64.b64encode(contents).decode('utf-8-sig')
    #        file.close()
    #        st.markdown(f'<img src="data:image/gif;base64,{data_url}>',unsafe_allow_html = True)
            
    #    else:
    #        st.warning(f"Image not found at {gif_path}")
    
    prompt=st.text_input("or Enter a question here")
    
    if prompt_selection:
    
        prompt = prompt_selection
    
    #audio_container = st.container()
    
    if prompt:
        start=time.process_time()
        response=retrieval_chain.invoke({"input":prompt})
        #print("Response time :",time.process_time()-start)
        st.write(response['answer'])    
        
        audio_path = sanitize_filename(f"{prompt}.mp3")
        #st.write(audio_path)
        text_audio_convert(client,response['answer'],audio_path)
        
        
        #st.audio(audio_path)
        
        
        #audio_file = open(audio_path, "rb")
        #audio_bytes = audio_file.read()
        #audio_play = st.audio(audio_path)
        st.audio(audio_path)
        
        #audio_container.empty()
        #audio_container = st.container()
        #with audio_container:
        #    with open(audio_path, "rb") as f:
        #        audio_bytes = f.read()
        #    base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        #    audio_html = f"""
        #    <audio controls autoplay>
        #    <source src="data:audio/mp3;base64,{base64_audio}" type="audio/mp3">
        #    </audio>
        #    """
        #    st.markdown(audio_html, unsafe_allow_html=True)
    
        # With a streamlit expander
        with st.expander("Document Sources"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")




















