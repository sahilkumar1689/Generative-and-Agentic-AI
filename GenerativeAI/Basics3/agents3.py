import os
import requests
from rich import print
from typing import Optional, List
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.tools import tool, ToolRuntime
from langchain_groq import ChatGroq
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver  # In-memory persistence

load_dotenv()

openweather_api_key = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------------------------------------------------------
# 1. DEFINE STATE AND CONTEXT
# ------------------------------------------------------------------

# Static request data (Passed during .invoke) = It is a read-only context that can be used to pass user-specific information, like user_id, auth_token, etc.
class UserContext(BaseModel):
    user_id: str

# Dynamic memory (Persists and updates during execution) = It is a mutable state that can be updated during the agent's execution.
class UserState(AgentState):
    preferred_city: Optional[str] = None
    searched_topics: List[str] = []

# ------------------------------------------------------------------
# 2. DEFINE TOOLS WITH RUNTIME MEMORY
# ------------------------------------------------------------------

@tool
def get_weather(
    city_name: Optional[str] = None,
    runtime: ToolRuntime[UserContext, UserState] = None # here we can access both the request context and the dynamic state
) -> Command | str:
    """Gets current weather. If city_name is not provided, uses the saved preferred city."""
    
    # Read state to check for stored city if non provided
    target_city = city_name or runtime.state.get("preferred_city")

    if not target_city:
        return Command(update={
            "messages": [
                ToolMessage(
                    "No city provided and no preferred city found in memory. Ask the user for their city.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        })

    url = f"http://api.openweathermap.org/data/2.5/weather?q={target_city}&appid={openweather_api_key}&units=metric"
    data = requests.get(url).json()

    if data.get("cod") != 200:
        return f"Could not retrieve weather data for {target_city}."

    desc = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    
    # Save target_city into global state as preferred_city = With the help of the Command object, we can update the state and send messages back to the agent in a single response.
    return Command(update={
        "preferred_city": target_city,
        "messages": [
            ToolMessage(
                f"The weather in {target_city} is {desc} with {temp}°C.",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

# ------------------------------------------------------------------
# 3. SETUP AGENT WITH MEMORY (CHECKPOINTER)
# ------------------------------------------------------------------

model = ChatGroq(model="openai/gpt-oss-120b", api_key=GROQ_API_KEY)

# Short-term memory checkpointer to remember conversations per thread
checkpointer = MemorySaver()

agent = create_agent(
    model=model,
    tools=[get_weather],
    state_schema=UserState,
    context_schema=UserContext,
    checkpointer=checkpointer
)

# ------------------------------------------------------------------
# 4. EXECUTION WITH CONFIGURATION
# ------------------------------------------------------------------

# thread_id separates conversations for different users/sessions = If you try to call the agent with the same thread_id, it will remember the previous state. If you use a different thread_id, it will start fresh.Very userfull in multi-user applications where you want to maintain separate conversations for each user.

config = {"configurable": {"thread_id": "session_1"}}
user_ctx = UserContext(user_id="usr_99")

# Run 1: User doesn't specify a city initially

print("Enter 0 to exit.")

while True:
    user_input = input("User: ")
    if user_input.strip() == "0":
        break

    res = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        context=user_ctx
    )

    print("Agent:", res["messages"][-1].content)

    print("\n--- End of Session ---\n")
    print("messages list:\n", res["messages"])


# # Run 2: User asks again without mentioning Tokyo — preferred_city is saved in State!
# res2 = agent.invoke(
#     {"messages": [{"role": "user", "content": "How's the weather standard today?"}]},
#     config=config,
#     context=user_ctx
# )
# print("Agent:", res2["messages"][-1].content)