from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,ChatPromptTemplate

review_template_str = """Use the following context to answer questions.
Be as detailed as possible, but don't make up any information
that's not from the context. If you don't know an answer, say
you don't know.

{context}
"""

system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str,
    )
)

human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)


class Reader():
    def __init__(self,tokenizer,model) -> None:
        self.rag_prompt_template = ChatPromptTemplate(input_variables=["context", "question"],messages=[system_prompt,human_prompt],)
        self.tokenizer = tokenizer
        self.model = model
    
    def build_reader_llm(self,model, tokenizer, task) : 
        try : 
            reader_llm = pipeline(
            model=model,
            tokenizer=tokenizer,
            task=task,
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=500,
            )
            return reader_llm
        except Exception as e : 
            print(f"Issue when trying to build reader llm. Exception : {e}")
    
    def get_rag_prompt_template (self) :
        return self.rag_prompt_template 
    def get_tokenizer (self) :
        return self.tokenizer 
    def get_model (self) :
        return self.model 
    """
    def get_prompt_in_chat_format (self) :
        return self.prompt_in_chat_format    
    """