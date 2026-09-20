import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from Projects.project1.old_way.tools import web_scrapping,web_search
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 1. llm client:

llm_model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0.5,
    timeout=30
    # max_output_tokens=1000,
)


# 2. Create search agent:

class search_web_result(BaseModel):
    fetched_information:str
    key_take_away:str
    source_link:str

def search_agent():
    search_agent_ref = create_agent(
        model=llm_model,
        tools=[web_search],
        system_prompt=(
            "You are a web search assistant. Use web_search first. "
            "In your final answer, include every source title, exact URL, "
            "and description returned by the tool. Never omit or change URLs."
            "Special note = Add all source URLs at the starting of the response.In a separate source url section.After that continue with your normal flow."
        )
    )
    return search_agent_ref


# 3. Create web scrapping assistant:

def web_scrapping_agent():
    scrapping_agent = create_agent(
        model = llm_model,
        tools=[web_scrapping],
        system_prompt=(
            "You are a web scrapping assistant.You provided with the web result content which also contains websites URLS."
        )
    )
    return scrapping_agent


# 4. Create a deep search summarize:

deep_search_prompt_template = ChatPromptTemplate.from_messages([
    ('system','You are an expert research writter,write a clear,structure ad insightful report for the provided information.'),
    ('human',"""

write a detailed research report on the topic below.

Topic: {topic}

Researched gathered:
{research}

Structure the report as:
-Introduction
-Key Finding (minimum 3 well explained points)
-conclusion
-Sources (lis all urls found in the research)

Be detailed,factual and professional.
""")
])

parser = StrOutputParser()

deep_search_summarizer = deep_search_prompt_template | llm_model | parser



# 5. Critic llm:

critic_prompt_template = ChatPromptTemplate.from_messages([
    ("system","You are a sharp and constructive research critic.Be honest and specific."),
    ("human","""

    Review the research report below and evaluate it strictly.

    Report:
    {report}

    Respond in this exact format:

    Score = x/10

    Strengths:
    - ...
    - ...

    Area to Improve:
    - ...
    - ...

    One line verdict

""")
])


critic_summarizer = critic_prompt_template | llm_model | parser
