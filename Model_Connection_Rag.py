import os
import requests
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from Vector_Data_Store_Embed import create_index_file

prompt_template = """Your give the answer to the medical questions. If you know the answer give the answer. Give Output in english

Context:
{context}

Question:
{question}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

api_key = "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"
model_id = "deepseek/deepseek-r1-distill-qwen-14b:free"
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)

faiss_index_path = "index"
def faiss_loading():
    if os.path.exists(f"{faiss_index_path}.faiss"):
        try:
            db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            retriever = db.as_retriever(search_kwargs={"k": 3})
            return retriever
        except Exception as e:
            index_file=create_index_file()

    else:
        index_file=create_index_file()
    db=FAISS.load_local(faiss_index_path,embeddings,allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever

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
    except Exception as e:
        print("❌ API error:", e)
        return e

def question(query):
    try:
        retrivers=faiss_loading()
        retrieved_docs = retrivers.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        full_prompt = prompt.format(context=context, question=query)

        answer = query_openrouter(full_prompt)
    except Exception as e:
        return e

    return answer

    

