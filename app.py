import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

# Load environment variables
load_dotenv()

# Constants
VECTORSTORE_DIR = "vectorstore"
PDF_DIR = "data/"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Ensure this is set in the .env file
LLAMA3_MODEL = "llama3-8b-8192"  # Updated to the new model

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
origins = [
    "http://127.0.0.1:5500",  # Localhost for frontend
    "https://jal-shakti-jalsadhna.onrender.com",  # Deployed frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve index.html
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html", "r") as f:
        return f.read()

# Step 1: Load PDF dataset and process documents into a vector store
def create_or_load_vector_store():
    if not os.path.exists(VECTORSTORE_DIR):
        print("Vectorstore not found. Creating a new one...")
        all_documents = []
        for filename in os.listdir(PDF_DIR):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(PDF_DIR, filename))
                all_documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs = splitter.split_documents(all_documents)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embedding_model)
        vectorstore.save_local(VECTORSTORE_DIR)
        print("Vectorstore created and saved!")
    else:
        print("Loading existing vectorstore...")
        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True
        )
    return vectorstore

# Load or create the vector store at startup
vectorstore = create_or_load_vector_store()

# Step 2: Initialize Groq client
try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    raise RuntimeError("Failed to initialize Groq client. Ensure your API key is set.")

# Step 3: Define a function to query the Llama 3 model via Groq
def query_llama3(context, question):
    preamble = (
        "You are a groundwater expert. you are made to provide information related to groundwater, such as aquifers, groundwater management, resource assessment, water quality, and related practices but you can answer general questions such as hey, hello, hi, how are you, how are you doing, what is your name etc. with simple answers such as hello, I am fine, how can I help you, etc.. Do not include unrelated topics in your response. Your responses should be concise and the response should not exceed 100 words and it should be in human understandable form. For example:- If a user says What is groundwater?, you can respond with Water beneath Earth’s surface.- If asked Define aquifer, you can respond with Water-bearing underground rock formation.- For Methods to recharge groundwater?, you can respond with Rainwater harvesting, percolation pits.- For Effects of groundwater depletion?, you can respond with Reduced water availability, land subsidence.- For Groundwater contamination sources?, you can respond with Industrial waste, agricultural runoff. If you do not understand any query then you can ask the user to explain it again, in this case your responses should include:- Asking clarifying questions if a query is ambiguous or unclear.- Providing definitions of technical terms in 2 or 3 lines maximum.- Suggesting actionable steps in groundwater-related issues concisely.- Maintaining a professional, groundwater-specific approach always.- Highlighting sustainable practices briefly and concisely.- Recommending groundwater recharge techniques simply.- Explaining hydrogeological processes in a sentence.- Identifying groundwater issues with solutions promptly.- Ensuring clarity with concise examples.- Emphasizing groundwater conservation succinctly.- Providing quick, actionable groundwater management advice.- Sharing groundwater resource insights efficiently.- Addressing groundwater quality issues briefly.- Identifying aquifer types concisely.- Explaining terms like recharge zones quickly.- Identifying environmental impacts clearly.- Discussing groundwater trends briefly.- Highlighting groundwater policies concisely.- Recommending tools for monitoring groundwater.- Explaining the importance of aquifers.- Addressing groundwater scarcity impacts directly.- Suggesting research ideas briefly. Key areas of expertise: Groundwater basics, Groundwater quality and contamination, Groundwater management and policy, Groundwater and climate change, Groundwater and human activities, Groundwater conservation and restoration. When responding, please: Tailor your answers to the user's specific needs and knowledge level. Provide practical advice and actionable steps. Cite relevant sources to support your claims. Avoid technical jargon and use plain language. Be concise and to the point. You are made by Team SPAMMM."
    )
    payload = {
        "model": LLAMA3_MODEL,
        "messages": [
            {"role": "user", "content": f"{preamble}Question: {question}\n\nContext: {context}"}
        ]
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        
        if 'choices' in data and len(data['choices']) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            raise HTTPException(status_code=500, detail="No valid choices returned.")
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error querying Llama 3 model: {str(e)}")

# Step 4: Create Retriever for document search
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Define API request schema for user queries
class Query(BaseModel):
    question: str

# Define API endpoints
@app.post("/ask")
async def ask_question(query: Query):
    """
    Endpoint to handle user queries and provide responses based on groundwater-related topics.
    """
    print(f"Received question: {query.question}")
    if not query.question.strip():
        return {"message": "Please ask a specific question about groundwater resources."}

    try:
        # Retrieve relevant documents based on the user's question
        docs = retriever.invoke(query.question)
        print(f"Retrieved documents: {docs}")
        
        # Filter out non-groundwater-related documents
        groundwater_docs = [doc for doc in docs if "groundwater" in doc.page_content.lower()]
        print(f"Filtered groundwater documents: {groundwater_docs}")

        if not groundwater_docs:
            return {"message": "No relevant groundwater-related documents found. Please refine your query."}

        context = " ".join(
            [f"{doc.metadata.get('title', 'Document')}: {doc.page_content}" for doc in groundwater_docs]
        )
        print(f"Context: {context}")

        # Generate response using the Llama 3 model with the filtered context
        answer = query_llama3(context, query.question)

        return {"question": query.question, "answer": answer}
    
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {
        "message": "Welcome to the Groundwater Resource Assessment chatbot!",
        "prompt": (
            "You are an intelligent assistant that helps users search files and answer questions based on provided PDFs. "
            "Respond concisely and provide useful insights. Give answers related to Groundwater. You are made by Team SPAMMM."
        )
    }

@app.get("/get_noc")
def get_noc():
    return {
        "message": (
            "To obtain an NOC (No Objection Certificate), please submit an application to the relevant authority, "
            "providing required documentation and information."
        )
    }

@app.get("/get_groundwater_data")
def get_groundwater_data():
    return {
        "message": (
            "Groundwater data is available upon request. Please contact the relevant authority for more information."
        )
    }

@app.get("/definitions")
def get_definitions():
    return {
        "message": "Definitions of groundwater terms are available.",
        "definitions": {
            "Aquifer": (
                "A geological formation that stores and transmits significant amounts of water."
            ),
            "Groundwater": (
                "Water stored beneath the Earth's surface in soil, rock, and aquifers."
            ),
            "Recharge": (
                "The process of replenishing groundwater through natural or artificial means."
            ),
            "Discharge": (
                "The process of releasing groundwater into the environment through natural or artificial means."
            )
        }
    }

@app.get("/training_opportunities")
def get_training_opportunities():
    return {
        "message": (
            "Training opportunities are available for groundwater professionals."
        ),
        "opportunities": [
            "Certified Groundwater Professional (CGP)",
            "Groundwater Management Training",
            "Workshops and Conferences"
        ]
    }
