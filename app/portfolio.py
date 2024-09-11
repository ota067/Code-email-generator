from os import name
import uuid
import chromadb
import pandas as pd

class Portfolio():
    def __init__(self,file_path ="resourse/my_portfolio.csv") :
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.chroma_client = chromadb.PersistentClient('vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(name= "portfolio")

    def load_portfolio(self):
        if not self.collection.count(): # Corrected typo in 'count'
           for _, row in self.data.iterrows():
               self.collection.add(documents=row["Techstack"],
                   metadatas={"links":row["links"]}, # Added colon to create a dictionary
                   ids=[str(uuid.uuid4())])
    def query_link(self, skills):
        return  self.collection.query(query_texts=skills,n_results=2).get("metadatas",[])           

    

        