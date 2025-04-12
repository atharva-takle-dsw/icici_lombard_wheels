import os
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
import chainlit as cl
from openai import OpenAI
from typing import Dict, Optional
import json

from utils import get_context, get_response, EmbeddingsAPI

from configs.config import API_KEY, MODEL_NAME, COLLECTION_NAME, DB_PATH, OPENAI_API_BASE, EMBEDDING_ENDPOINT


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    '''
    func to authenticate user
    '''
    # Read the username and passwords
    with open('configs/default-userbase.json') as json_file:
        ALLOWED_USERS = json.load(json_file)
    if os.path.exists('configs/extra-userbase.json'): # For extra configured users
        with open('configs/extra-userbase.json') as json_file:
            ALLOWED_USERS.update(json.load(json_file))

    #compare the password with the value stored in the database
    if username in ALLOWED_USERS and ALLOWED_USERS[username]==password:
        # Catching the first name of user
        user_name = username.split("@")[0].split(".")[0].capitalize()
        return cl.User(
            identifier=user_name, metadata={"role": "USER", "provider": "credentials"}
        )
    else:
        return None

@cl.on_chat_start
async def on_chat_start():
    embeddings=EmbeddingsAPI(endpoint=EMBEDDING_ENDPOINT)
    icici_docsearch = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH
    )
    llm = ChatOpenAI(
        openai_api_base = OPENAI_API_BASE,
        api_key = API_KEY,
        model_name = MODEL_NAME 
    ) 
    # Store instances in user session correctly
    cl.user_session.set("icici_docsearch", icici_docsearch)
    cl.user_session.set("llm", llm)
    cl.user_session.set("chat_history", [])  # Store memory instance directly  
    await cl.Message(content="""Welcome! I am sales assistant of ICICI Lombard Insurance Company. I will help you with all your insurance needs.
 
Let's get you insured!
 
Please share your query to get started.""").send()


@cl.on_message
async def main(message: cl.Message):
    # Retrieve memory and Other variables
    chat_history = cl.user_session.get("chat_history")
    icici_docsearch = cl.user_session.get("icici_docsearch")

    llm = cl.user_session.get("llm")
    user_query = message.content

    loader_msg = cl.Message(content="")
    await loader_msg.send()
   
    context = await get_context(user_query, icici_docsearch)
    response = await get_response(user_query, context, llm, chat_history)
   
    print(f"Response received: {response}")
    await cl.Message(content=response).send()
    # Save user input and AI response using save_context method  
    chat_history.append({"User": message.content, "Assistant": response})
    
    cl.user_session.set("chat_history", chat_history)
    print(f"Current chat history: {chat_history}")