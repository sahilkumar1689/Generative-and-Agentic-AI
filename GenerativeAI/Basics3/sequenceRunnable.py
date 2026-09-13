import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Initialization:
prompt_template = ChatPromptTemplate.from_template("Explain {topic} in simple 500 words.")

parser = StrOutputParser()

model = ChatMistralAI(
    model="mistral-small",
    api_key=MISTRAL_API_KEY,
    temperature=0.7,
    timeout=30,
    max_tokens=1000
)



# General Usage:

# prompt = prompt_template.format(topic="RAG")

# result = model.invoke(prompt)

# formated_result = parser.parse(result.content)

# print("Formatted Result:\n",formated_result)


# Runnables:

chain = prompt_template | model | parser

result = chain.invoke("RAG")

print("Formated Result:\n",result)

