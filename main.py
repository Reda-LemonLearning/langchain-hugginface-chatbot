import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from huggingface_hub import login

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

question = "Who is the 2008 ballon d'or winner ?"
template = """Question: {question}
Answer : Let's think step by step.
"""
prompt = PromptTemplate.from_template(template)

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.run(question))