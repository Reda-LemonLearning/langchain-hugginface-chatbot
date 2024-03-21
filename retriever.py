from vector_database import Chroma

class Retriever():
    def __init__(self, data_path) -> None:
        self.data_path = data_path
        self.vector_db = None
    
    def get_vector_db(self,embedding_model) : 
        try : 
            documents_vector_db = Chroma(
            persist_directory=self.data_path,
            embedding_function=embedding_model)
            return documents_vector_db
        except Exception as e : 
            print(f"Issue when retrieving vector db. Exception : {e}")

    def get_documents_retriever(self,documents_vector_db,k) : 
        try : 
            documents_retriever  = documents_vector_db.as_retriever(k=k)
            return documents_retriever
        except Exception as e : 
            print(f"Issue when trying to setup documents retriever. Exception : {e}")