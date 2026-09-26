import os
from rich import print
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,ToolMessage
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")



# 1. Creating Tools:
@tool
def weather(name: str) -> str:
    """This is a tool that takes a city name and returns their current weather."""
    return f"The weather in {name} is sunny."

# You can invoke this tool:
# result = weather.invoke({"name": "New York"})
# print(result)  # Output: "The weather in New York is sunny."


# print("Tool name:\n",weather.name)
# print("Tool description:\n",weather.description)
# print("Tool parameters:\n",weather.args)


# 2. Tool Binding:


model = ChatGroq(
    model="qwen/qwen3.6-27b",
    api_key=GROQ_API_KEY,
    temperature=0.7,
    timeout=30
    # max_output_tokens=1000,
)

llm_with_tools = model.bind_tools([weather])

# result = llm_with_tools.invoke("Tell me the weather in Chicago.")

# print(result)



# 3. Prompting with Tools:

tool_dict = {
    "weather": weather
}

messages = []

user_prompt = input("You: ")

prompt = HumanMessage(user_prompt)
messages.append(prompt)

# print("After user prompt, messages list is:\n", messages)


modal_result = llm_with_tools.invoke(messages)

messages.append(modal_result)
# print("After model result, messages list is:\n", messages)

if modal_result.tool_calls:
    for tool_call in modal_result.tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")

        if tool_name in tool_dict:
            tool = tool_dict[tool_name]
            tool_result = tool.invoke(tool_args)
            messages.append(ToolMessage(content=tool_result,tool_call_id=tool_call.get("id","")))
                            
            print("After tool result, messages list is:\n", messages)
            # print(f"Tool '{tool_name}' result: {tool_result}")
        else:
            messages.append(f'No tool found with the name: {tool_name}')
else:
    print(f"Model response: {modal_result.content}")

# print("Final messages list is:\n", messages)
final_response = llm_with_tools.invoke(messages)
print(f"Final response: {final_response.content}")


