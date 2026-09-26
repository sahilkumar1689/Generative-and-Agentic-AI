import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage


load_dotenv()

MISTRAL_API_Key = os.getenv("MISTRAL_API_KEY")

chat_history = []

chat_model = ChatMistralAI(
    model="mistral-small",
    api_key=MISTRAL_API_Key,
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
)


system_message = input("Enter system prompt: ")
chat_history.append(SystemMessage(content=system_message))

print("\n Enter 0 to exit the chat. \n")

while True:
    user_input = input("User: ")
    chat_history.append(HumanMessage(content=user_input))

    if user_input == "0":
        print("Exiting the chat...")
        break

    response = chat_model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("Response : ",response.content)


print("\nChat History: \n")

print(chat_history)