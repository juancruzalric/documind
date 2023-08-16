import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
from langchain import LLMChain, HuggingFaceHub
from transformers import pipeline
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate

HUGGINGFACE_API_TOKEN= os.getenv("HUGGINFACEHUB__API_TOKEN")

def get_pdf_text(docs):
    text = ""
    for doc in docs:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(chunks):
  embeddings = OpenAIEmbeddings()
  #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
  vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
  return vectorstore

def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    #llm = HuggingFaceHub(repo_id="google/flan-t5-base",
    #                     model_kwargs={"temperature": 0, "max_length": 64},
    #                     huggingfacehub_api_token=HUGGINGFACE_API_TOKEN)

    llm = ChatOpenAI(model="gpt-4", temperature=0)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            
def main():

    load_dotenv()


    st.set_page_config(
        page_title="Documind: Chat with multiple files",
        page_icon=":books:"
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.write(css, unsafe_allow_html=True)
    st.header("Documind")
    st.header("Chat with your documents :books:")
    user_question = st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Your documents")
        files = st.file_uploader(
            "Upload your document here", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get the file text
                raw_text = get_pdf_text(files)
                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # Create vector store
                vectorstore = get_vectorstore(text_chunks)
                # Create conversation
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success('Documents processed!', icon="✅")

            
    if user_question:
        handle_user_input(user_question)
    


if __name__ == "__main__":
    main()