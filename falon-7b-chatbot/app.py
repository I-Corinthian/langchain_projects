import os 
from apikey import apikey
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint


#api
os.environ["HUGGINGFACEHUB_API_TOKEN"] = apikey

# model
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct",
    max_new_tokens=600,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)

#promt template
template = """You are a helpful AI assistant and provide the answer for the question based on the given context.
>>QUESTION<<{question}
>>ANSWER<<""".strip()
promt_template = PromptTemplate(input_variables=["question"],template=template)

#chain init
chain = LLMChain(
    llm=llm,
    prompt=promt_template,
)

#UI
st.title("Falcon 7b model ")
prompt = st.text_input("ask your question")

if prompt:
    response = chain.invoke(prompt)
    st.write(response.get("text"))
    