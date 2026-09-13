import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List,Optional
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate

# Load environment variables from .env file:
load_dotenv()

MISTRAL_API_Key = os.getenv("MISTRAL_API_KEY")


# Create the static thinks:
class UserInformation(BaseModel):
    name: str
    age: Optional[int]
    email: Optional[str]
    address: Optional[str]
    phone_number: Optional[str]
    hobbies: Optional[List[str]]
    skills: Optional[List[str]]
    education: Optional[List[str]]
    work_experience: Optional[List[str]]

parser = PydanticOutputParser(pydantic_object=UserInformation)


chat_modal = ChatMistralAI(
    model = "mistral-small",
    api_key = MISTRAL_API_Key,
    temperature = 0.7,
    timeout = 30,
    max_tokens = 1000
)




# Create prompt template:

# prompt_template = PromptTemplate.from_template(
#     """You are a helpful assistant. You will be given a user information and you need to extract the following information from it:{output_response_template} 
    
#     User Information: {user_information}
#     """
# )

prompt_template = ChatPromptTemplate.from_messages([
    ("system",""" You are a helpful assistant. You will be given a user information and you need to extract the following information from it:{output_response_template}"""),
    ("human","""User Information: {user_information}""")
])



user_info = """My name is John Doe. I am 30 years old. My email address is john.doe@example.com. I live at 123 Main Street, Anytown, USA. My phone number is (555) 123-4567. I enjoy hiking, reading, and playing the guitar. I have skills in Python programming, data analysis, and web development. I have a Bachelor's degree in Computer Science from XYZ University. I have worked as a software engineer at ABC Company for 5 years."""


# final_prompt = prompt_template.format(output_response_template=parser.get_format_instructions(), user_information=user_info)

final_prompt = prompt_template.invoke({"output_response_template": parser.get_format_instructions(), "user_information": user_info})




response = chat_modal.invoke(final_prompt)

parsed_response = parser.parse(response.content)

print("Parsed Response: \n", parsed_response)