from dotenv import load_dotenv
load_dotenv()
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def load_llm():
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                              temperature=0.5,
                              model_kwargs={"token":os.getenv("HF_TOKEN"),
                                            "max_tokens":"512"})
    return llm
    
    
PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(PROMPT_TEMPLATE):
    prompt = PromptTemplate(template=PROMPT_TEMPLATE,input_variables=["context","question"])
    return prompt

DB_PATH="embeddings/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)

query=RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(PROMPT_TEMPLATE)}
)

user_query=input("Write Query Here: ")
response=query.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])