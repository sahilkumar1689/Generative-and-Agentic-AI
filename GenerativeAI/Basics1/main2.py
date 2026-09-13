from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text1 = "My name is sahil kumar and I am a software engineer. I love to code and build applications."
text2 = "I live in Jalandhar, Punjab, India. But i worked in Ludhiana, Punjab, India. I have experience in building web applications and APIs using Python and JavaScript."
text3 = "India is a land of diversity and culture. It is known for its rich history, traditions, and festivals. The country has a diverse population with various languages, religions, and cuisines."

# 1. Used to see the embedings directly:
# embedding1 = embeddings.embed_query(text1)
# embedding2 = embeddings.embed_query(text2)

# embedding3 = embeddings.embed_documents([text1,text2])

# print("Embedding for text1: \n", embedding1)
# print("Embedding for text2: \n", embedding2)

# print("Embeddings for both texts: \n", embedding3)


# 2. Used to create and store the embedings as well:

vector_store = InMemoryVectorStore.from_texts([text1, text2, text3], embedding=embeddings)

# 3. Retrievel strategies:

# retriever = vector_store.as_retriever(search_kwargs={"k": 1}) # k = 1 means it will return the most similar document to the query.

# retriever = vector_store.as_retriever(search_type = "similarity_score_threshold",search_kwargs={"search_threshold": 0.5}) # it means it will return all the documents that have similarity score greater than 0.5 to the query.

retriever = vector_store.as_retriever(search_type = "mmr",search_kwargs={
        "k": 2,              # Number of final documents to return
        "fetch_k": 10,       # Initial pool of top relevant documents to evaluate
        "lambda_mult": 0.5   # 0.0 = max diversity, 1.0 = max relevance
    })

similarity_result2 = retriever.invoke("What is the software engineer name?")

print("Similarity Result 2: \n", similarity_result2) # by default if nothing meet the search_threshold it will return the all.

 
# Get the similarity score along with the document using similarity_search_with_score() method. It will return a list of tuples where each tuple contains the document and its similarity score.
# retriever_with_Score = vector_store.similarity_search_with_score("What is the software engineer name?")

# print("Similarity Result with Score: \n", retriever_with_Score)


