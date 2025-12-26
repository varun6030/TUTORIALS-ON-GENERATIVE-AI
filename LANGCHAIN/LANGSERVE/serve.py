import sys
sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
from dotenv import load_dotenv
import os

load_dotenv()

# Load API key
groq_api_key = os.getenv("GROQ_API_KEY")

# LLM
model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Translate the following into {language}:"),
        ("user", "{text}")
    ]
)

# Output parser
parser = StrOutputParser()

# Chain
chain = prompt | model | parser

# FastAPI app
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain runnable interface"
)

# Add LangServe routes
add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)


