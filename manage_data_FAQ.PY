import pandas as pd
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import dotenv
from langchain_neo4j import Neo4jGraph
from langgraph.graph import END
import ast

# Load environment variables
dotenv.load_dotenv(override=True)

# Neo4j connection
enhanced_graph = Neo4jGraph(enhanced_schema=True)

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dimensional embedding

# Load your CSV file
df = pd.read_csv('D:/Kl_project/chatbot_demo/Final_FAQ.csv')

# Extract the required columns
subset = df[['question', 'answer']].copy()

# Create a list to hold the embeddings
embeddings = []

# Loop over each row in the DataFrame
for index, row in subset.iterrows():
    question = row['question'] or ""
    answer = row['answer'] or ""
    text = f"{question}. {answer}"
    embedding = model.encode(text).tolist()
    embeddings.append(embedding)
    
    print(f"Q: {question}\nA: {answer}\nEmbedding: {embedding[:5]}...")  # Truncated for readability

# Add the embeddings as a new column
subset['embedding'] = embeddings

# Save to a new CSV file (or overwrite the existing one)
subset.to_csv('D:/Kl_project/chatbot_demo/Final_FAQ_with_embeddings.csv', index=False)

print("Embeddings saved to Final_FAQ_with_embeddings.csv")
