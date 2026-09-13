import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import ChatMistralAI



load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# 1. Load the pdf:

# pdf_loader = PyMuPDFLoader("data_Structure.pdf")
# loaded_data = pdf_loader.load() # slow for bg pdf's
# lazy_ref = pdf_loader.lazy_load()

# loaded_data = [doc for doc in lazy_ref]

# print("Loaded data:\n",loaded_data)

# 2. Text splitting:

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=200,
#     chunk_overlap=20
# )

# chunks = splitter.split_documents(loaded_data)

# print("Chunks:\n",chunks)


# 3. Embedding:

# embeddings = FastEmbedEmbeddings(
#     model_name="BAAI/bge-small-en-v1.5"
# )

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# print("Embeddings:\n",embeddings.embed_documents(chunks))

# 4. Vector Store:


# 1. User to inititalize the db and store chunks in a specific collections.If you call it multiple time it append the chunks in the same instance:

# doc1_ids = [f"data_structure_chunk_{i}" for i in range(len(chunks))]

# vector_store = Chroma.from_documents(
#     documents=chunks,
#     embedding=embeddings,
#     ids=doc1_ids,   # store ids to uniquely identify the chunks.So that we can perform CRUD operation on them.
#     persist_directory="./chroma_db",
#     collection_name="user_knowledge_base2"
# )


# 2. Used to retreive the existing db instance:
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db",
    collection_name="user_knowledge_base2"
)


# 3. After that you can also perform the crud operation on them:

# add_documents() = Used to append new pdf chunks in the specific collection.

# vector_store.add_documents(documents=more_chunks, ids=more_ids)

# delete() / delete_collection() = Used to delete the specific document on the basis of their meta_Data or ids.So set you meta_data and ids accordingly so that you can easily filter them.

# vector_store.delete(ids=doc1_ids)




# print("Vector store:\n",vetor_Store)

# 5. Retriever:

# retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k": 4})


# result = retriever.invoke("What is array?")

# print("Result:\n",result)

# for doc in result:
#     print("\n")
#     print(doc.page_content)
#     print("\n")

# 6. New retreiver strategy called MultiQuery Retriever:

retreiver = vector_store.as_retriever();

modal = ChatMistralAI(
    model="mistral-small",
    api_key=MISTRAL_API_KEY,
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
)

multi_retreiver = MultiQueryRetriever.from_llm(retriever=retreiver,llm=modal)

result = multi_retreiver.invoke("What is array?")

print("Result:\n",result)



