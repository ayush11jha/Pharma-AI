import streamlit as st
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
load_dotenv()

DB_FAISS_PATH="embeddings/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm():
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.5,
        task="text-generation",
        model_kwargs={"token":os.getenv("HF_TOKEN"),
                      "max_length":"512"}

    )
    return llm


def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Ask your query...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """

        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            source_documents=response["source_documents"]
            source_docs_formatted = ""
            if source_documents:
                source_docs_formatted += "\nSource Docs:\n"
                count = 1
                for doc in source_documents:
                    document_id = getattr(doc, 'id', 'No ID')
                    # print(f"Document metadata: {doc.metadata}")

                    if isinstance(doc.metadata, dict):
                        # source = doc.metadata.get('source', 'No Source')
                        page_label = doc.metadata.get('page_label', 'No Page Label')
                    else:
                        # source = 'No Source'
                        page_label = 'No Page Label'

                    source_docs_formatted += f"\nDocument ID: {document_id} , \tPage Label: {page_label}\n"
                    # source_docs_formatted += f"\nPage Label: {page_label}"

                    with st.expander(f"Show Source {count} Content"):
                        st.write(doc.page_content)
                    count += 1
            result_to_show=result + "\n\n" + source_docs_formatted
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()