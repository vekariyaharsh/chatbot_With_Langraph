from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

query_text="i want itam for my birth day party which item ypu can provide for that?"
embedding_vector = model.encode(query_text)
print(embedding_vector)

