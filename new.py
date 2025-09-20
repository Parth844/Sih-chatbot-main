import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load PDF dataset
pdf_dir = "data/"  # Directory containing PDF files
all_documents = []

# Iterate through all PDF files in the directory
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_dir, filename))
        all_documents.extend(loader.load())

# Step 2: Split data into manageable chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(all_documents)

# Step 3: Create vector embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)

# Step 4: Save the vectorstore for retrieval
vectorstore.save_local("vectorstore")
print("Vectorstore saved!")
