import os
from google import genai
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFacePipeline


load_dotenv()


# Access environment variables
Google_API_Key = os.getenv("GOOGLE_API_KEY")
GROQ_API_Key = os.getenv("GROQ_API_KEY")
MISTRAL_API_Key = os.getenv("MISTRAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_KEY")

# print("Google API Key loaded from .env file:", Google_API_Key)



#  1. CHECKING AVAILABLE MODELS:

# # Initialize the new genai Client
# client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# print("Available models:\n")
# for model in client.models.list():
#     print(f"Model Name: {model.name}")
#     print(f"Display Name: {model.display_name}")
#     print("-" * 40)



# 2. Initialize A Model:



# 1. Initialize the chat model with the specified parameters:

# Google Chat model:
# chat_model = init_chat_model(
#     "gemini-3.6-flash",
#     model_provider="google_genai",
#     api_key=Google_API_Key,
#     temperature=0.7,
#     timeout=30,
#     max_tokens=1000,
#     max_retries=3
# )


# Groq Chat model:
# chat_model = init_chat_model(
#     "llama-3.3-70b-versatile",
#     model_provider="groq",
#     api_key=GROQ_API_Key,
#     temperature=0.7,
#     max_tokens=1000,
# )

# Mistral Chat model:
# chat_model = init_chat_model(
#     "mistral-small",
#     model_provider="mistralai",
#     api_key=MISTRAL_API_Key,
#     temperature=0.7,
#     max_tokens=1000,
# )

# Openai Chat model:
# chat_model = init_chat_model(
#     "gpt-5.6",
#     model_provider="openai",
#     api_key=OPENAI_API_KEY,
#     temperature=0.7,
#     max_tokens=1000,
# )




# 2. Genearate using the Google GenAI Client (Alternative Approach):

# chat_model = ChatGoogleGenerativeAI(
#     model="gemini-3.6-flash",
#     api_key=os.getenv("GOOGLE_API_KEY"),
#     temperature=0.7,
#     timeout=30,
#     max_output_tokens=1000,
# )



# chat_model = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     api_key=GROQ_API_Key,
#     temperature=0.7,
#     timeout=30
#     # max_output_tokens=1000,
# )



# chat_model = ChatMistralAI(
#     model="mistral-small",
#     api_key=MISTRAL_API_Key,
#     temperature=0.7,
#     timeout=30,
#     max_tokens=1000,
# )


# chat_model = ChatOpenAI(
#     model="gpt-5.5",
#     api_key=OPENAI_API_KEY,
#     temperature=0.7,
#     timeout=30,
#     max_tokens=1000,
# )



# Call the llm with a prompt and get the response:

# 1. invoke() = This method is used to send a prompt to the model and receive a response. It takes a string input (the prompt) and returns a response object containing the generated text.It is Synchronous.
# response = chat_model.invoke("Hello, how are you today?")

# print("Response from the chat model:\n")
# print(response.content[0].get("text", ""))


# 2. pass the conversation list:
# from langchain.messages import HumanMessage, AIMessage, SystemMessage

# # conversation = [
# #     {"role": "system", "content": "You are a helpful assistant that translates English to French."},
# #     {"role": "user", "content": "Translate: I love programming."},
# #     {"role": "assistant", "content": "J'adore la programmation."},
# #     {"role": "user", "content": "Translate: I love building applications."}
# # ]

# conversation = [
#     SystemMessage("You are a helpful assistant that translates English to French."),
#     HumanMessage("Translate: I love programming."),
#     AIMessage("J'adore la programmation."),
#     HumanMessage("Translate: I love building applications.")
# ]

# response = chat_model.invoke(conversation)
# print(response.content[0].get("text", ""))  # AIMessage("J'adore créer des applications.")



# 3. stream() = This method is used to send a prompt to the model and receive a response in a streaming manner. It takes a string input (the prompt) and returns a generator that yields response chunks as they are generated. It is Asynchronous.

# full = None;

# for chunk in chat_model.stream("Hello"):
#     print(chunk)

#     # Every model stream result in different manner some has content as array or some has string so implement it acordingly.
#     if chunk and len(chunk.content) > 0:

#         chunk = chunk.content[0]
#         print(chunk)

        # if chunk.get("type") == "text":
        #     print(chunk.get("text", ""))
        #     full = full + chunk.get("text", "") if full else chunk.get("text", "")
        # else:
        #     print("\nError:", chunk.content[0].get("text", ""))
        #     break
  
    

# print("\n\nFull response received:\n")
# print(full)


# 4. Batch() = This method is used to send multiple prompts to the model in a single request and receive responses for each prompt. It takes a list of string inputs (the prompts) and returns a list of response objects containing the generated texts. It is Synchronous.

# batch_prompts = [
#     "Hello, how are you today?",
#     "What is the capital of France?",
#     "Tell me a joke."
# ]

# for i, response in enumerate(chat_model.batch(batch_prompts)):
#     print(f"Response {i + 1}")
#     print(f"Response: ",response)

# for i, response in enumerate(chat_model.batch_as_completed(batch_prompts)):
#     print(f"Response {i + 1}")
#     print(f"Response: ",response)




# 3. Generating Responses Using the Google GenAI Client (Direct Approach):

# client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


## print(client.models.generate_content(model="gemini-2.5-flash", contents="Hi").text)  # depricated

# interaction = client.interactions.create(
#     model="gemini-3.6-flash", input="Tell me a joke."
# )

# print(interaction.output_text)





# Hugging Face with langchain:

# llm_instance = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen3.8-2.4T-A95B",
#     temperature=0.5,
#     max_new_tokens=500)

# model = ChatHuggingFace(llm=llm_instance)

# response = model.invoke("Hey,how have you been?")

# print(response.content)


# Used to install the modal locally and then used it:

# llm = HuggingFacePipeline.from_model_id(
#     model_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     pipeline_kwargs=dict(
#         max_new_tokens=512,
#         do_sample=False,
#         repetition_penalty=1.03,
#     ),
# )

# chat_model = ChatHuggingFace(llm=llm)