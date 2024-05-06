import streamlit as st
import os
import timeit
from langchain.llms import CTransformers
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

import box
import yaml


# Import config vars
with open('config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def create_vector_db(loader):
    data = loader.load()
   
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP)
    text = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDINGS,
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(text, embeddings)
    db.save_local(cfg.DB_FAISS_PATH)    
    
def load_llm():
    llm = CTransformers(model=cfg.MODEL_BIN_PATH,
                        model_type=cfg.MODEL_TYPE,
                        max_new_tokens=cfg.MAX_NEW_TOKENS,
                        temperature=cfg.TEMPERATURE
    )
    return llm    

def retrieval_qa_chain():

    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDINGS,
                                       model_kwargs={'device': 'cpu'})
    db=FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)
    #retriever =db.as_retriever(score_threshold=0.7)
    retriever=db.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT})
    
    llm = load_llm()
    
    # Define the prompt template
    qa_template = """Use the following pieces of information to answer the user's question.
    Try to provide as much text as possible from "response". If you don't know the answer, please just say 
    "I don't know the answer". Don't try to make up an answer.
    
    Context: {context},
    Question: {question}
    
    Only return correct and helpful answer below and nothing else.
    Helpful answer: """

    PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "question"])
                                        
    chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=retriever,
                                           input_key="query",
                                           return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
                                           chain_type_kwargs={'prompt': PROMPT})
    return chain
