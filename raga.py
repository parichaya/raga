"""
Project: RAGA
A RAG-based LLM System (Retrieval-based Augmented Generation LLM System)
Author Name : Parichaya Chatterji
       Email: chatterjiparichay@gmail.com
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------
# 1. Load documents
# ---------------------------
loader = TextLoader("data.txt")
documents = loader.load()

# ---------------------------
# 2. Split into chunks
# ---------------------------
text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)
texts = [doc.page_content for doc in docs]

print(f"Loaded {len(texts)} chunks")

# ---------------------------
# 3. Create embeddings
# ---------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(texts)

print(f"Embeddings shape: {embeddings.shape}")

# ---------------------------
# 4. Store in FAISS
# ---------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print("FAISS index ready")

# ---------------------------
# 5. Query + retrieval
# ---------------------------
query = "How does RAG work?"
query_embedding = embedder.encode([query])

k = 3
distances, indices = index.search(np.array(query_embedding), k)

# Remove duplicate chunks (important)
retrieved_docs = list(dict.fromkeys([texts[i] for i in indices[0]]))

print("\n--- Retrieved Context ---")
for i, doc in enumerate(retrieved_docs):
    print(f"\nChunk {i+1}:\n{doc}")

# ---------------------------
# 6. Local LLM generation (fast + good quality)
# ---------------------------
model_name = "google/flan-t5-base"

# Load once
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Reduce context size for speed
context = "\n".join(retrieved_docs[:2])

prompt = f"""
Explain clearly how Retrieval-Augmented Generation (RAG) works.

Context:
{context}

Question:
{query}

Answer in 3-4 sentences:
"""

print("\n--- Generating Answer ---")

inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    min_length=30,
    do_sample=False
)

#answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().capitalize()

print("\n--- Final Answer ---")
print(answer)