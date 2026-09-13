import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter,TokenTextSplitter,RecursiveCharacterTextSplitter

load_dotenv()

# Load Document:

text_loader = TextLoader("dummy.txt")
loaded_text = text_loader.load()[0].page_content;

print("Loaded text:\n",loaded_text)


# Split the text:

# 1. CharacterTextSplitter() = Used to split the text using chunk length measured by the length of characters.By default it split text using "\n\n".

# How CharacterTextSplitter Works:
# When chunk_size=10 and separator=" ":

# Step 1 (Split): It splits the text by spaces into single words (["My", "name", "is", "sahil", "kumar", ...])-it won't split inside a word.

# Step 2 (Merge): It merges words left-to-right until adding the next word exceeds 10 characters.

# Step 3 (Fall Through): If a single unit (or word) is longer than 10 characters on its own, it cannot split it further. It emits the whole unit intact and logs a warning: Created a chunk of size X, which is longer than the specified 10.

# splitter = CharacterTextSplitter(
#     separator=" ", # default value
#     chunk_size=10,
#     chunk_overlap=0,
#     length_function=len,
#     is_separator_regex=False,
# )


# 2. TokenTextSplitter() = TokenTextSplitter works completely differently from CharacterTextSplitter. It converts your text into machine tokens (using tiktoken by default) and counts tokens, not raw characters or words.

# How Tokenization Works on dummy.txt:
# Tokens are not characters or whole words—they are word fragments processed by LLMs.
# Tokenize: The splitter converts the entire text string into an array of integer token IDs.
# " software engineer." =  [' software', ' engine', 'er', '.'] (4 tokens)
# Slice: It cuts the integer array directly every 10 tokens (since chunk_size=10).
# Decode: It converts each group of 10 token IDs back into string chunks.

# splitter = TokenTextSplitter(
#     chunk_size=10,
#     chunk_overlap=0
# )


# 3. RecursiveCharacterTextSplitter() = RecursiveCharacterTextSplitter takes a list of separators (by default: ["\n\n", "\n", " ", ""]) and tries them in order. It attempts to split on the largest separator first, and if a chunk is still larger than chunk_size, it recursively falls back to smaller separators for that specific piece.

# working:
# Try "\n\n": Splits into paragraphs. Paragraphs > 10 chars? Move to next separator.

# Try "\n": Splits into lines. Lines > 10 chars? Move to next separator.

# Try " ": Splits into words. It groups words up to 10 chars.

# Try "" (Fallback): If a single word is > 10 chars (e.g., "applications" - 12 chars), unlike CharacterTextSplitter which gave a warning, RecursiveCharacterTextSplitter drops to "" and splits the long word in half ("applicat", "ions").


splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0
)



# Now you have two methods for splitting:

chunks = splitter.split_text(loaded_text) # used to simply return the chunked text strings.
# chunks = splitter.create_documents([loaded_text]) # used to return the langchain document contains metadata and page_Content.

print("Extracted Chunks length:\n",len(chunks))
print("Extracted Chunks:\n",chunks)



# Note = We also have html,markdown,code files,etc splitter.Refer the langchain documentation for text splitting