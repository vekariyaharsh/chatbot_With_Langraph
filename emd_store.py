from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from operator import add
import dotenv
from langchain_neo4j import Neo4jGraph
from langgraph.graph import END

dotenv.load_dotenv(override=True)
enhanced_graph = Neo4jGraph(enhanced_schema=True)   
# Neo4j connection


# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dimensional embedding

# Step 1: Get all Products + values + category
def fetch_products(tx):
    query = """
    MATCH (c:Category)-[:HAS_PRODUCT]->(p:Product)
    OPTIONAL MATCH (p)-[:HAS_OPTION]->(:ValueOption)-[:HAS_VALUE]->(v:Value)
    RETURN DISTINCT id(p) AS pid, p.name AS title, p.description AS description,
           c.name AS category
    """
    return tx.run(query)

# Step 2: Store embedding back into node
def store_embedding(tx, pid, embedding):
    query = """
    MATCH (p:Product) WHERE id(p) = $pid
    SET p.embedding = $embedding
    """
    tx.run(query, pid=pid, embedding=embedding.tolist())

# Step 3: Traverse & embed

records = enhanced_graph.query("MATCH (c:Category)-[:HAS_PRODUCT]->(p:Product) OPTIONAL MATCH (p)-[:HAS_OPTION]->(:ValueOption)-[:HAS_VALUE]->(v:Value) RETURN DISTINCT id(p) AS pid, p.name AS title, p.description AS description,c.name AS category")

for record in records:
    product_id = record["pid"]
    
    title = record["title"] or ""
    description = record["description"] or ""
    category = record["category"] or ""

    text = f"{title}. {description}. Category: {category}."
    embedding = model.encode(text).tolist()  # Ensure it's a list

    # Step 3: Update the node in Neo4j
    update_query = """
MATCH (p:Product) WHERE id(p) = $id
SET p.embedding = $embedding
"""
    enhanced_graph.query(update_query, params={"id": product_id, "embedding": embedding})
    print( product_id ,"added")

print("✅ Product embeddings generated and stored in Neo4j.")
