import os
from typing import List
from dotenv import load_dotenv

from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "user_123"  # Prototype default collection



def load_llm() -> ChatGroq:
    """
    Initializes and returns the ChatGroq LLM instance.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=GROQ_API_KEY,
        temperature=0.3,
        timeout=60,
    )


def load_agent(
    llm: ChatGroq,
    tool_arr: List[BaseTool] | None = None,
    system_prompt: str = "You are a helpful assistant."
):
    if tool_arr is None:
        tool_arr = []

    return create_agent(
        model=llm,
        tools=tool_arr,
        system_prompt=system_prompt
    )



def load_embeddings() -> HuggingFaceEmbeddings:
    """
    Loads HuggingFace MiniLM sentence embeddings model.
    """
    try:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        raise RuntimeError(f"Failed to load HuggingFace Embeddings: {e}") from e


def split_transcript(transcript: str) -> List[str]:
    """
    Splits transcript into chunks suited for semantic retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    return splitter.split_text(transcript)


def create_vector_store(transcription: str) -> bool:
    """
    Splits the transcription into chunks, creates vector embeddings, and persists them into Chroma.
    """
    if not transcription or not transcription.strip():
        raise ValueError("Cannot create vector store from empty transcription.")

    print("Chunking transcript for embedding...")
    chunks = split_transcript(transcription)

    # Convert raw text strings to LangChain Document objects
    documents = [Document(page_content=chunk, metadata={"source": "transcript"}) for chunk in chunks]

    emb_model = load_embeddings()

    try:
        print(f"Creating/updating vector store in '{CHROMA_DB_DIR}'...")
        Chroma.from_documents(
            documents=documents,
            embedding=emb_model,
            persist_directory=CHROMA_DB_DIR,
            collection_name=COLLECTION_NAME,
        )
        print(f"Successfully stored {len(documents)} chunks in ChromaDB.")
        return True
    except Exception as e:
        raise RuntimeError(f"Error initializing Chroma vector store: {e}") from e


def get_vector_store() -> Chroma:
    """
    Loads an existing vector store instance from local storage.
    """
    emb_model = load_embeddings()
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=emb_model,
        collection_name=COLLECTION_NAME,
    )


def ask_question(retriever, rec_quest: List[str] = None) -> None:
    """
    Runs an interactive RAG chat session in the terminal.
    """
    llm_model = load_llm()
    parser = StrOutputParser()

    @tool
    def transcript_retriever(query: str) -> str:
        """Searches the audio transcript for relevant information."""

        docs = retriever.invoke(query)

        if not docs:
            return "No relevant transcript content was found."

        return "\n\n".join(doc.page_content for doc in docs)


    system_prompt = (
        "You are an intelligent assistant. You have access to a tool 'transcript_retriever' "
        "that searches an audio transcript. Use this tool when answering questions about the video content. "
        "For general follow-up questions, word explanations, or clarifying your previous responses, "
        "answer directly using the existing conversation history without calling the tool."
    )

    # Define RAG prompt template
#     chat_prompt_template = ChatPromptTemplate.from_messages([
#         (
#             "system",
#             """You are a helpful AI assistant. Answer the user's question using ONLY the provided transcript context below.
# If the context does not contain enough information to answer the question, state politely that you don't know based on the provided material.

# Context:
# {context}""",
#         ),
#         ("human", "Question: {question}"),
#     ])

    # chat_chain = chat_prompt_template | llm_model | parser

    # create_agent:
    agent = load_agent(llm_model,[transcript_retriever],system_prompt)

    print("\n" + "=" * 50)
    print("               RAG CHAT SESSION")
    print("=" * 50)

    if rec_quest:
        print("\nRecommended questions you can ask:")
        for idx, ques in enumerate(rec_quest, 1):
            print(f"  [{idx}] {ques}")

    print("\nType your question, enter a number (1-3) to select a recommended question, or type '0' or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input in ("0", "exit", "quit"):
                print("Exiting chat session. Goodbye!")
                break

            if not user_input:
                continue

            # Check if user typed a number corresponding to a recommended question
            if rec_quest and user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(rec_quest):
                    user_input = rec_quest[idx]
                    print(f"Selected Question: {user_input}")
                else:
                    print(f"Invalid option. Please select a number between 1 and {len(rec_quest)}.")
                    continue

            # Retrieve relevant chunks from ChromaDB
            # matched_docs = retriever.invoke(user_input)
            # if not matched_docs:
            #     print("Bot: No relevant context found in the transcript.\n")
            #     continue

            # context = "\n\n".join([doc.page_content for doc in matched_docs])

            # Generate answer using RAG chain
            # print("Bot is thinking...")
            # answer = chat_chain.invoke({"context": context, "question": user_input})


            # print(f"\nBot: {answer}\n" + "-" * 50)


            # Append current user prompt to history
            # messages.append()

            print("Bot is thinking...")
            result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})

            # Retrieve updated message state returned by the agent graph
            messages = result["messages"]
            bot_reply = messages[-1].content

            print(f"\nBot: {bot_reply}\n" + "-" * 50)



        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nError processing query: {e}\n")


def start_chat(rec_quest: List[str] = None) -> None:
    """
    Initializes vector store retriever and launches the chat loop.
    """
    try:
        vector_store = get_vector_store()
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        ask_question(retriever, rec_quest)
    except Exception as e:
        print(f"Failed to start chat: {e}")