# FastAPI Application
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import datetime
import asyncio

app = FastAPI()

# Import your Python script's functionalities
from test3 import send_to_llm  # Replace with the actual import


# Model for chat requests
class ChatRequest(BaseModel):
    text: str

# Model for chat responses
class ChatResponse(BaseModel):
    response: str
    timestamp: datetime.datetime

# Endpoint to handle chat requests
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Call the LLM function and handle the response
    llm_response, _ = await send_to_llm(request.text)
    if llm_response is None:
        llm_response = "Error: No response from LLM."  # Default error message

    return ChatResponse(response=llm_response, timestamp=datetime.datetime.now())

# Mute functionality placeholder
is_muted = False


@app.get("/mute")
def mute():
    global is_muted
    is_muted = not is_muted
    return {"muted": is_muted}

@app.get("/")
def root():
    return {"message": "FastAPI ChatGLM3-6B Server Running"}

# Ensure that the send_to_llm function in your Python script is correctly implemented
# and can communicate with your LLM model effectively.
