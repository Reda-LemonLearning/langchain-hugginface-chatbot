import os
from vector_database import VectorDatabase
from retriever import Retriever
from reader import Reader
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import torch

def answer_with_rag(question: str,
    llm,
    knowledge_index,
    prompt_template,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 5,
):
    # Gather documents with retriever
    print("=> Retrieving documents...")
    relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text
    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

    final_prompt = prompt_template.format(question=question, context=context)

    # Redact an answer
    print("=> Generating answer...")
    answer = llm(final_prompt)[0]["generated_text"]

    return answer, relevant_docs

question = "How to create a pipeline object ?"

def get_answer(vector_db_retriever,reader, model, question) : 
    documents_chain = (
    {"context": vector_db_retriever, "question": RunnablePassthrough()}
    | reader.get_rag_prompt_template()
    | model
    | StrOutputParser())
    answer = documents_chain.invoke(question)
    return answer


if __name__ == '__main__':   
    path = "chroma_data"

    MODEL_REPO_ID = os.getenv("MODEL_REPO_ID")

    print("Loading models\n")
    model = AutoModelForCausalLM.from_pretrained(MODEL_REPO_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
    print("Finished loading models\n")

    print("Loading embedding model\n")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
    )
    print("Finished loading embedding model\n")
    print("Building vector db\n")
    vector_retriever_builder = Retriever(path)
    vector_db = vector_retriever_builder.get_vector_db(embedding_model)
    vector_db_retriever = vector_retriever_builder.get_documents_retriever(vector_db,10)

    print("Finished building vector db\n")
    print("Building reader\n")
    reader_builder = Reader(tokenizer,model)
    prompt_template = reader_builder.get_rag_prompt_template()
    reader = reader_builder.build_reader_llm(model,tokenizer,"text-generation")
    print("Finished building reader\n")

    print("Trying to get answer\n")
    answer = answer_with_rag(question,model,vector_db,reader,prompt_template)
    print(answer)