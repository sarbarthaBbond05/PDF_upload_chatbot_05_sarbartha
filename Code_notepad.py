import pdfplumber    #reason for using pdfplumber is that it is a library that allows us to extract text from PDF files.
import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.header("Sarbartha_chatbot")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

with st.sidebar:
    st.title("Upload your file")
    #file_1 = st.file_uploader("Upload your file", type=["pdf", "docx", "txt"])
    file_1 = st.file_uploader("Upload your file", type="pdf")

if file_1 is not None:
    with pdfplumber.open(file_1) as in_pdf:
        text = ""
        for page in in_pdf.pages:
            text += page.extract_text() + "\n"
    #st.write(text)

    #split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
        );
    chunks = text_splitter.split_text(text)
    #st.write(chunks)

    #Embedding the chunks
    embeddings = GoogleGenerativeAIEmbeddings(
    #model="models/text-embedding-004",
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
    )

    #Create a vector store from the embeddings
    vector_store = FAISS.from_texts(chunks, embeddings)

    #collecting user query
    user_question = st.text_input("Ask a question about the document:")

    #generate answer
    #question -> embeddings -> similiairty search -> results to LLM -> response (CHAIN)

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    retriever = vector_store.as_retriever(
        search_type="mmr",    #signify that we want to use MMR (Maximal Marginal Relevance) for retrieval
        search_kwargs={"k":4} #Use MMR to get more diverse results, if we use k=4, it will return 4 results
    )

    #define the LLM and prompts
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,    #if we give higher temperature, it will give more creative answers, but less accurate answers
        max_output_tokens=1000, #signify the maximum number of tokens we want in the output, 
                                #if we give higher value, it will give more detailed answers, more time to generate the answer
        google_api_key=GEMINI_API_KEY
    )

    #define the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions about a PDF document.\n\n"
         "Guidelines:\n"
         "1. Provide complete, well-explained answers using the context below.\n"
         "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
         "3. If the context mentions related information, include it to give fuller picture.\n"
         "4. Only use information from the provided context - do not use outside knowledge.\n"
         "5. Summarize long information, ideally in bullets where needed\n"
         "6. If the information is not in the context, say so politely.\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    if user_question:
        response = chain.invoke(user_question)
        st.write(response)

else:
    st.write("Please upload a PDF file to get started.")


