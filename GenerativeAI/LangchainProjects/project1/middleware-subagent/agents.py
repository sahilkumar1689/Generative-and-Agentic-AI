import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from tools import web_search, web_scrapping

load_dotenv()

# Set temperature to 0.0 for deterministic JSON tool calls
llm_model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0,
    timeout=60
)

# -------------------------------------------------------------
# PYDANTIC INPUT SCHEMAS FOR SUB-AGENTS
# -------------------------------------------------------------

class ResearchInput(BaseModel):
    query: str = Field(description="The search topic or detailed question to research.")

class ScrapingInput(BaseModel):
    url: str = Field(description="The target website URL to scrape.")

class CriticInput(BaseModel):
    report: str = Field(description="The full research report text to evaluate.")

# -------------------------------------------------------------
# SUB-AGENT TOOLS WITH SCHEMAS
# -------------------------------------------------------------

@tool("research_subagent_tool", args_schema=ResearchInput)
def research_subagent_tool(query: str) -> str:
    """Gathers factual details and web search data for a given research topic."""
    agent = create_agent(
        model=llm_model,
        tools=[web_search],
        system_prompt="You are a search assistant. Use `web_search` with a clear search query."
    )
    res = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return res["messages"][-1].content


@tool("scraping_subagent_tool", args_schema=ScrapingInput)
def scraping_subagent_tool(url: str) -> str:
    """Extracts main text content from a specified web page URL."""
    agent = create_agent(
        model=llm_model,
        tools=[web_scrapping],
        system_prompt="Extract main text content from the given URL."
    )
    res = agent.invoke({"messages": [{"role": "user", "content": url}]})
    return res["messages"][-1].content


@tool("critic_evaluator_tool", args_schema=CriticInput)
def critic_evaluator_tool(report: str) -> str:
    """Evaluates the report quality and assigns a score out of 10."""
    prompt = (
        "Evaluate the draft report below.\n"
        "Return output in this format:\n"
        "Score: X/10\n"
        "Strengths:\n - ...\n"
        "Weaknesses:\n - ...\n"
    )
    truncated_report = report[:3000] if len(report) > 3000 else report
    res = llm_model.invoke(f"{prompt}\n\nReport:\n{truncated_report}")
    return res.content

# -------------------------------------------------------------
# SUPERVISOR AGENT
# -------------------------------------------------------------

def create_deep_search_supervisor():
    supervisor_prompt = """You are a Lead Research Supervisor.
    
    When asked to research a topic:
    1. Call `research_subagent_tool` with a clear search query string (e.g. {"query": "iphone 18 pro specs"}).
    2. Optional: Call `scraping_subagent_tool` with a URL (e.g. {"url": "https://..."}) if needed.
    3. Draft a research report.
    4. Call `critic_evaluator_tool` with your draft (e.g. {"report": "draft report text"}).
    5. If the critic score is 8/10 or higher, output your final report to the user.
    """

    supervisor = create_agent(
        model=llm_model,
        tools=[research_subagent_tool, scraping_subagent_tool, critic_evaluator_tool],
        system_prompt=supervisor_prompt
    )
    return supervisor