import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader,PyPDFLoader,WebBaseLoader

load_dotenv()


# 1. Load a text file

text_loader = TextLoader("text.txt")

# 2. Load a PDF file
pdf_loader = PyPDFLoader("document.pdf")



# 3. Load a web page

urls = ["https://wikipedia.org/wiki/Artificial_intelligence"]

web_loader = WebBaseLoader(urls)




# 4. Load the complete text from the file:

# text_documents = text_loader.load()

# print("Text file loaded successfully!")
# print(text_documents[0].page_content)


# 5. Load lazily (load only when needed):

# text_loader_lazy = text_loader.lazy_load()
# text_loader_lazy = pdf_loader.lazy_load()
text_loader_lazy = web_loader.lazy_load()

# print("Count: ",len(text_loader_lazy))
count = 0

for doc in text_loader_lazy:
    print("Lazy loaded PDF document:")
    print(doc.page_content)
    count += 1

print("Count: ",count)