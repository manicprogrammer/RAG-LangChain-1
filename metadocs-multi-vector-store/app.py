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

    # I found that if I parse off a single \n it doesn't seem to work so well. I find that perplexing and get
    # the same poor results if I setup the document as \n after every line versus the example test docs as they are 
    # provieded with \n\n. More experimentation is needed to understand why this is the case.
    split_data = string_data.split("\n\n")

    vectorstore = FAISS.from_texts(split_data, embedding=embedding)
    retriever = vectorstore.as_retriever()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    # have been using "Who played a major role in defending the ukraine ? Explain to me in 5 lines"
    # and what are the 10 principles of SSI?
    question = st.text_input("Input your question for the uploaded document")

    result = chain.invoke(question)

    st.write(result)


