import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chain import Chain
from utils import clean_text
from portfolio import Portfolio
from langchain_groq import ChatGroq

def creat_app(llm,Portfolio,clean_text):
    st.title("code mail generator")
    url_input = st.text_input("enter a url",value="https://jobs.nike.com/fr/")
    submit_button = st.button("submit")

    if submit_button :
        loader= WebBaseLoader([url_input])
        data = clean_text(loader.load().pop().page_content)
        Portfolio.load_portfolio()
        jobs = Chain.extracte_job(data, clean_text)
        for job in jobs:
           skills = job.get('skills',[])
           links = Portfolio.query_link(skills)
           email = llm.write_email(job,links)
           st.code(email, language='markdown')
          

if __name__ == "__main__":
    llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key="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"  # Ensure this is correctly set in your .env file
    )

    chain = Chain(llm)
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    creat_app(chain, portfolio, clean_text)           

