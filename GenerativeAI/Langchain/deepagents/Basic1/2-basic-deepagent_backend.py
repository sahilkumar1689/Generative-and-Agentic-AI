import os
from rich import print
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.types import Command
from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from deepagents import create_deep_agent, FilesystemPermission
from deepagents.backends import CompositeBackend, StateBackend,FilesystemBackend,StoreBackend

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# print(f"GROQ_API_KEY: {GROQ_API_KEY}")

# Initialize the Groq client:
model = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    temperature=0.7,
    timeout=30
    # max_output_tokens=1000,
)




# Deepagent Backends = Every deep agent has a virtual filesystem and their file tool used that virtual filesystem to read and write files. But the backend decides where the data is stored and how long it is available. There are three types of backends available in deepagents:

# 1. StateBackend: This backend store data in langgraph in-memory state and data available within single thread only.
# The data is present inside the agent in-memory 'files' dictinonary and we must need to carried forward that state to keep using them.
#  Also if we don't specify any backend then by default StateBackend will be used.
#  It is suitable for scenarios where we need to work with the temporary working files which are available only for the current session and not needed to be persisted across sessions.


# state_agent = create_deep_agent(
#     model = model,
#     backend = StateBackend(),
#     checkpointer = MemorySaver()  # without this conversation history will not be saved and we will lose the context of the conversation in next iteration.
# )
# config = {"configurable": {"thread_id": "session_1"}}

# result = None

# while True:
#     user_input = input("You: ")
#     if user_input.strip() == "0":
#         break

#     result = state_agent.invoke({
#         "messages": [{
#             "role": "user",
#             "content": user_input
#         }]
#     },config=config)

#     # print(result)
#     print("bot: ", result.get("messages")[-1].content)

# print("Result:\n",result)





# 2. FilesystemBackend: This backend store data in local filesystem and data available across multiple threads and sessions. It is suitable for scenarios where we need to work with the files which are needed to be persisted across sessions and can be shared across multiple threads.

# file_agent = create_deep_agent(
#     model = model, 
#     backend = FilesystemBackend(root_dir=".",virtual_mode=True), 
#     checkpointer = MemorySaver(),
#     interrupt_on = {
#         "delete": True  # Interrupt on delete operation
#     }
#     )


# 1. You can add the read , write permission with the filesystem backend.
# permissions = [
#     FilesystemPermission(
#         operations=["write"], # it can be "read", "write" or "read,write". write permission includes the create,update and delete operations. read permission includes the read operation.
#         path=["."],
#         mode="interrupt"  # it can be "interrupt" , "allow" or "deny". If it is "interrupt" then the agent will ask for human approval before performing the operation. If it is "allow" then the agent will perform the operation without asking for human approval. If it is "deny" then the agent will not perform the operation and will return an error message.
#     )
# ]

# Note = 1. You can define the multiple permissions but keep that in mind langgraph follow the first match wins policy.so ordering of the permissions is important.
# 2. Because write permisson includes the create,update and delete operations as well.So if you want to apply the interrupts on custom opeartion then you can use the "interrupt_on" parameter instead of "permissions" parameter 


# config = {"configurable": {"thread_id": "session_1"}}



# result = None

# while True:

#     user_input = input("You: ")

#     if user_input.strip() == "0":
#         break

#     result = file_agent.invoke(
#         {
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": user_input
#                 }
#             ]
#         },
#         config=config
#     )


#     # -----------------------------------------------------
#     # Check whether the agent was interrupted
#     # -----------------------------------------------------

#     if "__interrupt__" in result:

#         interrupt_info = result["__interrupt__"]

#         print("\n HUMAN APPROVAL REQUIRED")
#         print(interrupt_info)


#         # -------------------------------------------------
#         # Ask the human
#         # -------------------------------------------------

#         approval = input(
#             "\nDo you want to allow this delete operation? "
#             "(yes/no): "
#         ).strip().lower()


#         if approval == "yes":

#             # Resume the SAME interrupted execution
#             result = file_agent.invoke(
#                 Command(
#                     resume={
#                         "decisions": [
#                             {
#                                 "type": "approve"
#                             }
#                         ]
#                     }
#                 ),
#                 config=config
#             )

#             print(" Delete operation approved.")


#         else:

#             # Reject the tool call
#             result = file_agent.invoke(
#                 Command(
#                     resume={
#                         "decisions": [
#                             {
#                                 "type": "reject"
#                             }
#                         ]
#                     }
#                 ),
#                 config=config
#             )

#             print(" Delete operation rejected.")

#     print(
#         "bot: ",
#         result.get("messages")[-1].content
#     )


# print("Result:\n", result)



# 3. StoreBackend = This backend store data in langgraph store and data available across mutiple threads and sessions. It is suitable for scenarios where we need to work with the files which are needed to be persisted across sessions and can be shared across multiple threads.

# store = InMemoryStore()  # behave as a backend for the storebackend where the data stores. For production you can use the database.

# store_agent = create_deep_agent(
#     model = model,
#     backend = StoreBackend(namespace=lambda x:("user","user1")),
#     checkpointer = MemorySaver(),
#     store = store
# )

# config = {"configurable": {"thread_id": "session_1"}}

# result = None

# while True:
#     user_input = input("You: ")

#     if user_input.strip() == "0":
#         break

#     result = store_agent.invoke({
#         "messages": [{
#             "role": "user",
#             "content": user_input
#         }]
#     },config=config)

#     print("bot: ", result.get("messages")[-1].content)

# print("Result:\n",result)




# 4. CompositeBackend = In this we can combine multiple backends together and a single deep agent route those requests to the appropriate backend based on the matched configure path.

agent = create_deep_agent(
    model="google_genai:gemini-3.6-flash",
    backend=CompositeBackend(
        default=StateBackend(),  # for all request that does not match any of the configured routes, use StateBackend as the default backend
        routes={
            "/memories/": StoreBackend(
                namespace=lambda rt: (rt.server_info.user.identity,),
            ),
            "/policies/": StoreBackend(
                namespace=lambda rt: (rt.context.org_id,),
            ),
        },
    ),
)