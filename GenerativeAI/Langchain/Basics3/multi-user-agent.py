import os
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel

from langchain.tools import tool, ToolRuntime
from langchain_groq import ChatGroq
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

# ------------------------------------------------------------------
# 1. SCHEMAS: Context (Request level) & State (Thread memory level)
# ------------------------------------------------------------------

class RequestContext(BaseModel):
    user_id: str
    auth_token: str

class AgentUserState(AgentState):
    preferred_city: Optional[str] = None
    search_count: int = 0

# ------------------------------------------------------------------
# 2. STATEFUL TOOL DEFINITION
# ------------------------------------------------------------------

@tool
def get_weather(
    city_name: Optional[str] = None,
    runtime: ToolRuntime[RequestContext, AgentUserState] = None
) -> Command | str:
    """Gets weather. Uses stored preference if city name is omitted."""
    
    # Read user context (who is calling)
    active_user = runtime.context.user_id
    
    # Read dynamic memory (what has been saved in this thread)
    target_city = city_name or runtime.state.get("preferred_city")
    
    if not target_city:
        return Command(update={
            "messages": [ToolMessage("City not provided or found in memory.", tool_call_id=runtime.tool_call_id)]
        })

    # Update thread state counter and preferred city
    current_count = runtime.state.get("search_count", 0) + 1
    
    # (Mock API fetch for illustration)
    weather_info = f"Weather in {target_city}: 24°C, Sunny (User {active_user} lookups: {current_count})"

    return Command(update={
        "preferred_city": target_city,
        "search_count": current_count,
        "messages": [ToolMessage(weather_info, tool_call_id=runtime.tool_call_id)]
    })

# ------------------------------------------------------------------
# 3. GLOBAL AGENT INITIALIZATION & DATABASE CONNECTION POOL
# ------------------------------------------------------------------

DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/agent_db")
connection_pool = AsyncConnectionPool(conninfo=DB_URI, max_size=20, open=False)

# One global model and agent instance for the entire application server
model = ChatGroq(model="openai/gpt-oss-120b")
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    await connection_pool.open()
    
    # Initialize Postgres checkpointer
    checkpointer = AsyncPostgresSaver(connection_pool)
    await checkpointer.setup()  # Creates checkpointer database tables automatically
    
    # Build generic agent attached to checkpointer
    agent = create_agent(
        model=model,
        tools=[get_weather],
        state_schema=AgentUserState,
        context_schema=RequestContext,
        checkpointer=checkpointer
    )
    yield
    await connection_pool.close()

app = FastAPI(lifespan=lifespan)

# ------------------------------------------------------------------
# 4. MULTI-TENANT API ENDPOINT
# ------------------------------------------------------------------

class ChatPayload(BaseModel):
    thread_id: str   # Unique conversation session ID from frontend
    message: str     # Prompt content

@app.post("/api/chat")
async def chat_endpoint(
    payload: ChatPayload,
    x_user_id: str = Header(...),       # Extracted from authenticated user session
    authorization: str = Header(...)    # JWT token from frontend request header
):
    if not agent:
        raise HTTPException(status_code=500, detail="Agent uninitialized")

    # 1. Map session scope to config
    config = {
        "configurable": {
            "thread_id": f"user_{x_user_id}_thread_{payload.thread_id}"  # Prefixed for multi-tenant isolation
        }
    }
    
    # 2. Map request metadata to context
    request_ctx = RequestContext(
        user_id=x_user_id,
        auth_token=authorization
    )

    # 3. Invoke shared agent instance asynchronously
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": payload.message}]},
        config=config,
        context=request_ctx
    )

    return {
        "response": result["messages"][-1].content,
        "thread_id": payload.thread_id
    }