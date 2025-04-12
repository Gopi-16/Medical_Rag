import os
import requests
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

faiss_index_path="index"
document_path="data"

embedding_model="sentence-transformers/all-MiniLM-L6-v2"
embeddings=SentenceTransformerEmbeddings(model_name=embedding_model)
def create_index_file():
	try:
		documents=[]
		for root,_,files in os.walk(document_path):
			for file in files:
				if file.endswith(".pdf"):
					pdf_path=os.path.join(root,file)
					loader=PyMuPDFLoader(pdf_path)
					doc=loader.load()
					documents.extend(doc)
			if not documents:
				return "No Documnets"
		text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
		split_documents=text_splitter.split_documents(documents)		
		db=FAISS.from_documents(split_documents,embeddings)
		db.save_local(faiss_index_path)
		return db
	except Exception as e:
		return e

