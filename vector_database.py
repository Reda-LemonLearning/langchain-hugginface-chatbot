import pandas as pd
import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_community.vectorstores import Chroma

pd.set_option("display.max_colwidth", None)  # helpful when visualizing retriever outputs

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]


class VectorDatabase : 
    def __init__(self,vector_db_path):
        self.vector_db_path = vector_db_path

    def get_database_path(self) : 
        return self.vector_db_path

    def save_vector_database(self,embedding_model,docs_processed) : 
        try :
            documents_vector_db = Chroma.from_documents(docs_processed, embedding_model, persist_directory=self.vector_db_path)
            return documents_vector_db
        except Exception as e: 
            print(f"Issue when trying to save vector database : {e}")

    def get_dataset(self,path) : 
        try : 
            ds = datasets.load_dataset(path, split="train")
            raw_knowledge_base = [
        Document(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in ds
        ]
            return raw_knowledge_base
        except Exception as e : 
            print(f"Issue when trying to retrieve dataset : {path} \n Exception : {e}\n")

    def process_dataset(self,chunk_size,raw_knowledge_base,tokenizer_name):
        try : 
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(tokenizer_name),
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
            )
        except Exception as e :
            print(f"Error while trying to generate text splitter using pretrained model : {tokenizer_name}\n Exception : {e}\n") 

        try : 
            docs_processed = []
            for doc in raw_knowledge_base:
                docs_processed += text_splitter.split_documents([doc])
        except Exception as e : 
            print(f"Error while trying to split the knowlegde base documents. Exception : {e}\n")
        
        try : 
            unique_texts = {}
            docs_processed_unique = []
            for doc in docs_processed:
                if doc.page_content not in unique_texts:
                    unique_texts[doc.page_content] = True
                    docs_processed_unique.append(doc)

            return docs_processed_unique
        
        except Exception as e : 
            print(f"Error when trying to remove duplicates from documents. Exception : {e}\n")
        pass