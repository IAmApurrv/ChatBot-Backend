import os
from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

assistantTemplate = f"""
You are an AI assistant chatbot named "GenAI".
Your expertise is exclusively in providing information and advice about anything.
Chat History: {{chatHistory}}
Question: {{question}}
Answer:
"""

assistantPromptTemplate = PromptTemplate(
    input_variables=["chatHistory", "question"],
    template=assistantTemplate
)

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# llm = ChatGroq(model="llama-3.1-8b-instant")
llm = ChatGroq(model="llama-3.3-70b-versatile")
# llm = ChatGroq(model="llama-3.2-3b-preview")
# llm = ChatGroq(model="llama-3.2-90b-text-preview")

conversationMemory = ConversationBufferMemory(
    memory_key="chatHistory", max_len=50, return_messages=True)

llm_chain = LLMChain(llm=llm, prompt=assistantPromptTemplate,
                     memory=conversationMemory)


class ChatMessage(BaseModel):
    prompt: str


@app.post("/ai/chat")
async def chat(chatMessage: ChatMessage):
    """Handle incoming messages and return responses."""
    # print(chat_message)
    try:
        chatHistory = conversationMemory.load_memory_variables({})[
            'chatHistory']

        response = await llm_chain.acall({"chatHistory": chatHistory, "question": chatMessage.prompt})

        conversationMemory.save_context({"user": chatMessage.prompt}, {
                                         "assistant": response["text"]})
        # return response["text"]
        return JSONResponse(content={"generation": response["text"]})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# uvicorn ChatBot:app --reload

if __name__ == "__main__":
    uvicorn.run("ChatBot:app", host="0.0.0.0", port=8000, reload=True)
# python ChatBot.py


"""
Hello! I am Apurrv. Can you introduce yourself?
What can you do?
What are some tips for studying effectively?
What is my name ?
"""
