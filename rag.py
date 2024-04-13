import streamlit as st
import os
import chromadb
from langchain.vectorstores.chroma import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
#from langchain.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import DirectoryLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import requests
import base64
import traceback
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from PIL import Image
import shutil
#!pip install pypdf, langchain, chromadb, sentence-transformers
#!pip install --upgrade --quiet  langchain-google-genai pillow
from google.cloud import storage
GOOGLE_API_KEY = "AIzaSyBc3QhhSZbhfaqOAponIDb3SVF91h7eubE"
## LLM model
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

st.set_page_config(page_title="WHO Communicator", page_icon="rag.png")

CLOUD_STORAGE_BUCKET = 'rag_app_embedded_docs'


def download_blob(file_name):
    gcs = storage.Client()
    bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)
    blob = bucket.blob(file_name)
    os.remove(file_name )
    blob.download_to_filename(file_name)
    shutil.rmtree(chroma_dir, ignore_errors=True)
    shutil.unpack_archive("chroma_db.zip", chroma_dir)

chroma_dir = './chroma_db'
if os.path.exists(chroma_dir):
    pass
else:
    shutil.unpack_archive("chroma_db.zip", ".")
filter_list = []
lang_list = ['English', 'Japanese', 'Korean', 'Arabic', 'Bahasa Indonesia', 'Bengali', 'Bulgarian', 'Chinese', 'Croatian', 'Czech', \
'Danish', 'Dutch', 'Estonian',  'Finnish', 'French', 'German', 'Gujarati', 'Greek', 'Hebrew', 'Hindi', 'Hungarian', 'Italian', \
'Kannada', 'Latvian', 'Lithuanian', 'Malayalam', 'Marathi', 'Norwegian', 'Polish', 'Portuguese', 'Romanian', 'Russian', 'Serbian', 'Slovak', \
'Slovenian', 'Spanish', 'Swahili', 'Swedish'
, 'Tamil', 'Telugu', 'Thai', 'Turkish', 'Ukrainian',  'Vietnamese']

#Make a retrieval object
client = chromadb.PersistentClient(path=chroma_dir)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(client=client, embedding_function=embedding_function, collection_name="articles_embeddings")
doc_list = list(set( [ meta['source'] for meta  in vectordb.get()['metadatas'] ]))
#st.write(f"Number Documents available for answering questions: { len(doc_list)}")


for doc in doc_list:
    filter_list.append({"source": doc})



# Sidebar contents
with st.sidebar:
    col1, col2  = st.columns([.25,.75])
    with col1:
        image = Image.open('rag.png')
        st.image(image)
    with col2:
        st.markdown("<h3 style='text-align: left'>WHO Communicator : Questions Answer With References </h3>", unsafe_allow_html= True)
        st.write("This application uses Gemini Model. It can get the answers from a repository of documents.")
    col1, col2, col3  = st.columns([.30,.30, .40])
    with col1:
        appmode = st.radio("**App Mode**", ['All Docs', 'Single Doc'], index=0, help="You may get answer from all documents or single document using Apllication Mode.")
    with col2:
        lang= st.selectbox("**Answer Language**", lang_list, index=0, help="Select a language to get the answer.")
    with col3:
        num_chunks = st.number_input('**Number of Chunks**', min_value=1, max_value=5, value=3,step=1, help="Number of chunks to be used to answer.")
      
    
  
    st.write("You can ask questions for topic Obesity e.g. How to prevent childhood obesity? \n \
    You can choose output language for answer. Default is English. You may select number of chunks to be used to answer the question. \
    You may select all documents or single document to get the answer.")

    if appmode == 'All Docs':
        default_list = None
        st.markdown(f"**Documnets available for answering questions:**")
        doc_names_text= ""
        for i, doc in enumerate(doc_list):
            doc_names_text += f"{i+1}. {doc} \n"
        st.write(doc_names_text)
        #st.text_area(doc_names_text, height=200, help="List of documents available for answering questions.")
    else:
        doc_names = st.selectbox("Select Document", doc_list, help="Select a document to get the answer.")



    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("For questions or feedback, please reach out to us, Email: team_name@gmail.com")

    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("Copyright: © 2024 WHO Communicator. All rights reserved.")

    
    

if appmode == 'All Docs':
    retriever = vectordb.as_retriever(search_kwargs={"k": num_chunks })
else:
    retriever = vectordb.as_retriever(search_kwargs={"k": num_chunks , "filter": {'source': doc_names}})
    st.write(f"Selected document to get the answer: {doc_names}. Only this document will be used to answer the question.")

#Define the prompt template and retrieval chain
template = """
You are a helpful AI assistant.
Answer based on the context provided. 
context: {context}
input: {input}
answer:
"""
prompt = PromptTemplate.from_template(template)
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

question = st.text_input("**Type your question below.**",  value="", help="Type your question here. Example: How to prevent childhood obesity?" )
col1, col2, col3, col4 = st.columns([.25,.30, .30, .15])
with col1:
    show_sources = st.checkbox("Show Sources", value=True, help="Show source document and page numbers used to generate the answer.")
with col2:
    print_text= st.checkbox("Show Text Chunks ", value=False, help="Print text chunks, used to answer the question.") 
with col3:
    refresh_rep = st.button("Refresh Repository", help="Click only if new reposotory is available.")
with col4:
    submit_button = st.button(label='SUBMIT', help='Click to submit question and get answer.')

if refresh_rep:
        try:
            download_blob("chroma_db.zip")
            st.write("Chroma DB Repository downloaded successfully.")
        except Exception as e:
            st.write(f"Error in downloading Chroma DB Repository: {e}")


if submit_button:
    try:
        st.markdown(f"**QUESTION:** {question}")
        llm_response = retrieval_chain.invoke({"input":question})
        ans = llm_response['answer']
        st.markdown(f"**ANSWER:**")
        if lang == 'English':
            st.write(ans)
        else:
            try:
                translated_ans = llm.invoke(f"Translate the following text to {lang} :\n {ans}")
                st.write(translated_ans.content)
            except Exception as e:
                st.write(f"Answer could not be translated to {lang}. See English answer below.")
                st.write(ans)
            
   
        if show_sources:
            st.markdown(f"**Sources - File name(s) and page number(s) used to generate answer:**")
            for i, source in enumerate(llm_response["context"]):
                st.write(f"{i+1}. {source.metadata['source']}    Page {source.metadata['page']}" )
        if print_text:
            st.markdown(f"**Text chunks used to generate answer:**" )
            for i, source in enumerate(llm_response["context"]):
                st.write(f"*From File {source.metadata['source']}    Page: {source.metadata['page'] : }*" )
                st.write(source.page_content)
    except Exception as e:
        st.write(f"Error in using LLM: {e}")
     

