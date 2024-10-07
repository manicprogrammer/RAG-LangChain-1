from dotenv import load_dotenv
# Import the Streamlit library
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

# should have a .env with a key OPENAI_API_KEY
load_dotenv()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
# don't have to provide the API key if you have a env variable of OPENAI_API_KEY as that is the default used
model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
embedding = OpenAIEmbeddings()

uploaded_file = st.file_uploader("Choose a txt file", type="txt")
if uploaded_file is not None:
    string_data = uploaded_file.getvalue().decode("utf-8")

    split_data = string_data.split("\n\n")

    vectorstore = FAISS.from_texts(split_data, embedding=embedding)
    retriever = vectorstore.as_retriever()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    question = st.text_input("Input your question for the uploaded document")

    result = chain.invoke(question)

    st.write(result)


