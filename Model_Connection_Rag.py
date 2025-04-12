import os
import requests
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from Vector_Data_Store_Embed import create_index_file
import streamlit as st
# Load environment variables


api_key = st.secrets["OPENROUTER_API_KEY"]
if not api_key:
    raise ValueError("API key not found. Make sure it's set in the .env file as OPENROUTER_API_KEY")

# API and model details
model_id = "deepseek/deepseek-r1-distill-qwen-32b:free"
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Embedding model
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
faiss_index_path = "index"

# Prompt template
prompt_template = """
Your job is to answer medical questions. If you know the answer, provide it. Only respond in English.

Context:
{context}

Question:
{question}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

def faiss_loading():
    """Load or create FAISS index and return a retriever."""
    if os.path.exists(f"{faiss_index_path}.faiss"):
        try:
            db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            retriever = db.as_retriever(search_kwargs={"k": 3})
            return retriever
        except Exception as e:
            print("Error loading FAISS index, recreating...", e)
            create_index_file()
    else:
        print("Index not found, creating...")
        create_index_file()
    db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 3})

def query_openrouter(prompt_text):
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt_text}]
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print("❌ API request failed:", e)
        return "API request failed: " + str(e)
    except KeyError:
        print("❌ Unexpected API response format:", response.text)
        return "Unexpected API response format."

def question(query):
    try:
        retriever = faiss_loading()
        retrieved_docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        full_prompt = prompt.format(context=context, question=query)
        answer = query_openrouter(full_prompt)
        return answer
    except Exception as e:
        print("❌ Error in processing question:", e)
        return f"Error: {str(e)}"

# Example usage:
# print(question("What are the symptoms of diabetes?"))
