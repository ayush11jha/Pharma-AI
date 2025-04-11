# PharmaAI - Medical RAG Chatbot

**PharmaAI** is an AI-powered **Retrieval-Augmented Generation (RAG) chatbot** designed to assist with medical queries. It utilizes a combination of **document retrieval** and **natural language generation** to provide relevant and reliable responses based on medical literature and resources.

## 🔥 Features

- **Context-Aware Responses**: Uses RAG to fetch relevant medical data before generating answers.
- **PDF Knowledge Integration**: Processes and extracts information from medical PDFs.
- **Embeddings-Based Search**: Uses embeddings to match queries with relevant medical documents.
- **Streamlit Interface**: Simple and interactive UI for user-friendly interaction.
- **GitHub-Hosted Data**: Medical PDFs and embeddings are stored in a GitHub repository for easy access.

## 🛠️ Tech Stack


- **Python**
- **Streamlit** (for UI)
- **FAISS** (for efficient similarity search)
- **Hugging Face Transformers** (for NLP processing)
- **Sentence Transformers** (for embeddings generation)
- **GitHub Storage** (for embedding and document access)

## 📌 Installation

Clone the repository:

```bash
git clone  https://github.com/ayush11jha/Pharma-AI.git
cd PharmaAI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run main.py
```

## 📁 Data Structure

- **/data**: Contains medical PDFs.
- **/embeddings**: Stores precomputed embeddings for efficient retrieval.
- **main.py**: Main script to launch the chatbot.
- **ingest.py/connect.py**: Helper functions for document processing and retrieval.

## 🚀 Deployment

For **Streamlit Cloud** deployment:

1. Push the code to a GitHub repository.
2. Connect the repo to Streamlit Cloud.
3. Ensure the app can access the embeddings via a direct URL or cloud storage.

## 🎯 Usage

1. Enter your medical query.
2. PharmaAI will fetch relevant information and generate an AI-assisted response.

## 🔮 Future Improvements

- Integration with real-time medical databases.
- Improved response accuracy using fine-tuned medical models.
- Advanced UI with chatbot memory and feedback mechanisms.

---

### 📢 Contribute

If you'd like to contribute:

1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Commit your changes and open a PR.



