import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# 1. Define Pydantic schema for structured output
class SummaryOutput(BaseModel):
    title: str = Field(
        description="A concise title for the transcript summary, strictly between 8 to 10 characters long."
    )
    final_summary: str = Field(
        description="A comprehensive, well-structured final summary combining all key points of the transcript."
    )
    recommended_questions: List[str] = Field(
        description="Exactly 3 engaging follow-up questions a user might ask based on this summary."
    )


def load_llm():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    return ChatGroq(
        model="openai/gpt-oss-120b",  # Standard Groq model name
        api_key=GROQ_API_KEY,
        temperature=0.3,
        timeout=60,
    )


def split_transcript(transcript: str) -> List[str]:
    """
    Splits long transcripts into manageable chunks for map-reduce style processing.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_text(transcript)


def summarize_transcript(transcript: str) -> dict:
    """
    Summarizes a full audio transcript and returns structured data containing:
    - title (8-10 chars)
    - final_summary
    - recommended_questions (list of 3 questions)
    """
    if not transcript or not transcript.strip():
        raise ValueError("Provided transcript is empty.")

    llm_model = load_llm()
    str_parser = StrOutputParser()
    pydantic_parser = PydanticOutputParser(pydantic_object=SummaryOutput)

    # 1. Partial Chunk Summarizer Prompt
    partial_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert transcript analyzer. Provide a concise summary capturing key information from this portion of the transcript.",
        ),
        ("human", "Transcript Portion:\n{partition}"),
    ])

    partial_chain = partial_prompt | llm_model | str_parser

    # Split transcript into sensible chunk sizes
    split_arr = split_transcript(transcript)
    partial_summaries = []

    print(f"Splitting transcript into {len(split_arr)} chunk(s) for partial processing...")

    # Summarize in chunks if long, or process directly if short
    if len(split_arr) > 1:
        for idx, partition in enumerate(split_arr):
            try:
                print(f"Summarizing chunk {idx + 1}/{len(split_arr)}...")
                summarized_chunk = partial_chain.invoke({"partition": partition})
                if summarized_chunk and summarized_chunk.strip():
                    partial_summaries.append(summarized_chunk.strip())
            except Exception as e:
                print(f"Warning: Failed to summarize chunk {idx + 1}: {e}")

        combined_partial = "\n\n".join(partial_summaries)
    else:
        combined_partial = transcript

    # 2. Final Consolidation & Structured Output Prompt
    final_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert content strategist and summarizer.
Combine the provided text into a single cohesive summary.

Strict Rules:
1. 'title': Must be a catchy title between 8 and 10 characters long (including spaces).
2. 'final_summary': A clear, professional, and detailed summary.
3. 'recommended_questions': A list of EXACTLY 3 relevant questions a user might ask about this topic who's answers are present in that summary.

{format_instructions}""",
        ),
        ("human", "Input Content:\n{partial_summarized_transcript}"),
    ])

    final_chain = final_prompt | llm_model | pydantic_parser

    try:
        print("Generating final structured summary, title, and recommended questions...")
        result: SummaryOutput = final_chain.invoke({
            "partial_summarized_transcript": combined_partial,
            "format_instructions": pydantic_parser.get_format_instructions(),
        })

        return result.model_dump()

    except Exception as e:
        raise RuntimeError(f"Failed to generate structured summary from LLM: {e}") from e