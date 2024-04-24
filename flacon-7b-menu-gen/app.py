import os
from apikey import apikey
from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint #model
from langchain.prompts import PromptTemplate #promt templ
from langchain.chains.llm import LLMChain  #LLM chain
from langchain.chains.sequential import SequentialChain

os.environ["HUGGINGFACEHUB_API_TOKEN"] = apikey

# model
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct",
    max_new_tokens=600,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.1,
)

# prompt template
prompt_template_name = PromptTemplate(
    input_variables=['cuisine'],
    template="i want to open a restaurent for {cuisine} food. Suggest a fancy name for this."
)

# chain
name_chain = LLMChain(
    llm=llm,
    prompt=prompt_template_name,
    output_key="restaurant_name"
)

# prompt template
prompt_template_menu = PromptTemplate(
    input_variables=['restaurant_name'],
    template="Suggest some menu items for {restaurant_name}. Return it ad a commaa sepatated list"
)

# chain
item_chain = LLMChain(
    llm=llm,
    prompt=prompt_template_menu,
    output_key="menu items"
)

chain = SequentialChain(
    chains=[name_chain,item_chain],
    input_variables=["cuisine"],
    output_variables=['restaurant_name','menu items']
)

def get_response(cuisine):
    response = chain.invoke(cuisine)
    return response