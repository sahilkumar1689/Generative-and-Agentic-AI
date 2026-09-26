import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnablePassthrough

load_dotenv()
MISTRAL_API_Key = os.getenv("MISTRAL_API_Key")

# 1. Creating the initalization:

short_prompt_template = ChatPromptTemplate.from_template(""" Explain {topic} in just 1-2 lines. """)
long_prompt_template = ChatPromptTemplate.from_template(""" Explain {topic} in just 10-15 lines. """)

model = ChatMistralAI(
    model="mistral-small",
    api_key=MISTRAL_API_Key,
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
)

parser = StrOutputParser()


# 2. Parallel Runnables:

# 1. You can send one input to all parallel runnables.
# chains = RunnableParallel(
#     {
#         "short_chain": short_prompt_template | model | parser,
#         "long_chain": long_prompt_template | model | parser
#     }
# )

# results = chains.invoke({"topic":"LLM"});

# print("Result:\n",results.get("short_chain"));
# print("Result:\n",results.get("long_chain"));


# 2. RunnableLambda() = You can also send sepearate input to individual prompt:

# chains = RunnableParallel(
#     {
#         "short_chain": RunnableLambda(lambda obj:obj.get('short_chain',"Joke")) | short_prompt_template | model | parser,
#         "long_chain": RunnableLambda(lambda obj:obj.get('long_chain', "Joke")) | long_prompt_template | model | parser
#     }
# ) 

# results = chains.invoke({
#     "short_chain":{"topic":"llm"},
#     "long_chain":{"topic":"RAG"}
# })

# print("Result:\n",results.get("short_chain"))
# print("Result:\n",results.get("long_chain"))


# 3. RunnablePassthrough() = You can return the midway output:

code_prompt = ChatPromptTemplate.from_messages(
    [("system","You are a software engineer AI assistant who help in writing code."),
    ("human","{topic}")
    ]
)

explanation_prompt = ChatPromptTemplate.from_messages(
    [("system","You are a software engineer AI assistant who help in explaining code."),
    ("human",'explain this code to me:\n {code}')
    ]
)


seq1 = code_prompt | model | parser

seq2 = RunnableParallel(
    {
        "code": RunnablePassthrough(),
        "explaination": explanation_prompt | model | parser
    }
)

chain = seq1 | seq2

results = chain.invoke({"topic":"Write a code to reverse a string in python."})

print("Results:\n",results.get("code"))
print("Results:\n",results.get("explaination"))