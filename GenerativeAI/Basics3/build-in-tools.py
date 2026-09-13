import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


load_dotenv()

Tavily_API_Key = os.getenv("TAVILY_API_Key")
MISTRAL_API_Key = os.getenv("MISTRAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# 1. Initalizing the things:

search_tool = TavilySearch(max_results=1, api_key=Tavily_API_Key)

# print("Tool name:",search_tool.name)
# print("Tool description:",search_tool.description)
# print("Tool parameters:",search_tool.args)



prompt = ChatPromptTemplate.from_template(""" 
You are a helpful assistant.

Summarize the news in simple bullet points.
{news}
""")

# model = ChatMistralAI(
#     model="mistral-small",
#     api_key=MISTRAL_API_Key,
#     temperature=0.7,
#     timeout=30,
#     max_tokens=1000,
# )
    
model = ChatGroq(
    model="groq/compound",
    api_key=GROQ_API_KEY,
    temperature=0.7,
    timeout=30
    # max_output_tokens=1000,
)

parser = StrOutputParser()


# 2. Calling:


chain = (
    search_tool 
    | RunnableLambda(lambda results: {"news": results}) 
    | prompt 
    | model 
    | parser
)


result = chain.invoke({"query":"Latest news on AI of this week."})

print("Result:\n",result)










