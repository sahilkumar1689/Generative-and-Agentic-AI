import os
import requests
from rich import print
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
openweather_api_key = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# 1. Creating Tools:

@tool
def get_weather(city_name:str)->str:
    """This is a tool that takes a city name and returns their current weather details."""

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={openweather_api_key}&units=metric"
    response = requests.get(url)
    data = response.json()

    if data["cod"] != 200:
        return f"Could not retrieve weather data for {city_name}. Please check the city name."

    # print("OpenWeatherMap Data:", data)
    weather_description = data["weather"][0]["description"]
    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]

    return f"The weather in {city_name} is {weather_description} with a temperature of {temperature}°C and humidity of {humidity}%."

@tool
def get_current_news(user_query:str)->str:
    """ This is a tool that takes a user query and returns the news related to that query."""

    tavily_query = user_query

    search_tool = TavilySearch(max_results=1, api_key=tavily_api_key)

    result = search_tool.invoke({"query": tavily_query})
    # print("Tavily Search Result:", result)

    news_list = list()

    if "results" in result and len(result["results"]) > 0:
        for news_item in result["results"]:
            news_headline = news_item.get("title", "No title available")
            news_url = news_item.get("url", "No URL available")
            news_content = news_item.get("content", "No content available")
            news_list.append(f"Headline:\n {news_headline}.\n Read more at: {news_url}.\n Description: {news_content}")

        return f"Current news headlines:\n {'\n\n'.join(news_list)}"
    else:
        return f"No news headlines found."

# Custom middleware:

@wrap_tool_call
def human_approval(request,handler):
    # Print the tool call request
    print("[bold green]Tool Call Request:[/bold green]", request)

    # Ask for human approval
    approval = input("Do you approve this tool call? (yes/no): ").strip().lower()

    if approval == "yes":
        # If approved, proceed with the tool call
        response = handler(request)
        print("[bold blue]Tool Call Response:[/bold blue]", response)
        return response
    else:
        # If not approved, return a message indicating rejection
        print("[bold red]Tool call rejected by human.[/bold red]")
        return {"error": "Tool call rejected by human."}


# 2. Create a autonomous agent using create_agents:

model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0.7,
    timeout=30
    # max_output_tokens=1000,
)

agent = create_agent(
    model= model,
    tools=[get_weather, get_current_news],
    system_prompt = "You are a helpful assistant. Be concise and accurate",
    middleware=[human_approval]
)

print("press 0 to exit.")

while True:
        
    user_prompt = input("You: ")
    if user_prompt.strip() == "0":
        break

    result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
    # print("Agent Result:\n", result)

    print("chat: ",result.get("messages")[-1].content)
