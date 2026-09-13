import os
import requests
from rich import print
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_tavily import TavilySearch

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

    # Useful TavilySearch arguments for customizing results:
    # max_results: Number of search results to return.
    # topic: Search category, such as "general", "news", or "finance".
    # search_depth: Search quality, usually "basic" or "advanced".
    # include_answer: Include Tavily's generated answer in the response.
    # include_raw_content: Include the extracted full page content.
    # include_images: Include relevant image URLs.
    # include_image_descriptions: Include descriptions for returned images.
    # time_range: Limit results to a period such as "day", "week", "month", or "year".
    # days: For news searches, limit results to the specified number of recent days.
    # include_domains: Only search the specified domains, for example ["bbc.com"].
    # exclude_domains: Ignore results from the specified domains.
    # country: Prioritize results from a particular country.
    # auto_parameters: Let Tavily automatically choose suitable search parameters.

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


# response = get_current_news.invoke({"user_query": "India vs Japan historic match sports news."})
# print("Current news headlines:\n", response)



# 2. Create a hard coded agent:

model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0.7,
    timeout=30
    # max_output_tokens=1000,
)

llm_with_tools = model.bind_tools([get_weather, get_current_news])


messages = list()
tool_dict = {
    "get_weather": get_weather,
    "get_current_news": get_current_news
}

print("press 0 to exit.")

while True:

    user_prompt = input("You: ")
    if user_prompt.strip() == "0":
        break

    messages.append(HumanMessage(user_prompt))

    llm_response = llm_with_tools.invoke(messages)
    messages.append(llm_response)

    if llm_response.tool_calls:
        for tool_call in llm_response.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args")

            if tool_name in tool_dict:
                result = tool_dict[tool_name].invoke(tool_args)
                messages.append(ToolMessage(content=result, tool_call_id=tool_call.get("id", "")))
                # print(result)
            else:
                messages.append(f'No tool found with the name: {tool_name}')

        final_response = llm_with_tools.invoke(messages)
        messages.append(final_response)
        print("chat: ", final_response.content)
    else:
        print("chat: ", llm_response.content)


print("messages list:", messages)

