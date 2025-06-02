from operator import add
from typing import Annotated, List
import dotenv
from typing_extensions import TypedDict
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Annotated
from langgraph.graph import END
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from neo4j.exceptions import CypherSyntaxError
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from sentence_transformers import SentenceTransformer
from langgraph.graph import END, START, StateGraph

model = SentenceTransformer("all-MiniLM-L6-v2")

from sentence_transformers import SentenceTransformer
import numpy as np




def embed_text(text):
    return model.encode(text).tolist()
dotenv.load_dotenv(override=True)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

enhanced_graph = Neo4jGraph(enhanced_schema=True)            



enhanced_grap = '''
    Schema:
    (:Category) - A type of grouping like "Hats", "Coaster", or "Table Covers" ,"Mouse Pad" . Fields: id, name, description, status.

    (:Product) - A product for sale. Fields: id, name, description, status. In Product there are multiple type of product for one cetory like for table cover cetory product are Custom Rectangle Table Covers,Custom Square Table Cover, 

    (:Option) - A customization category, Product  (e.g. size, color,Velcro Clip,Cover Style,Side, Height,Thickness, Material,Runner Size). Fields: id, name. In this option are avelable for Product and Category .

    (:OptionValue) - A value of an option (e.g. Small, Blue, size, color,Velcro Clip,Cover Style,Side, Height,Thickness, Material,Runner Size). Fields: id, name.

    Relationships:
    (Category)-[:HAS_PRODUCT]->(Product)
    (Product)-[:HAS_OPTION]->(Option)
    (Option)-[:HAS_VALUE]->(OptionValue)
'''

class InputState(TypedDict):
    question: str


class OverallState(TypedDict):
    question: str
    next_action: str
    cypher_statement: str
    cypher_errors: List[str]
    database_records: List[dict]
    steps: Annotated[List[str], add]
    history: List[Dict[str, str]]
    related_products: List[str]


class OutputState(TypedDict):
    question: str
    answer: str
    steps: List[str]
    cypher_statement: str


def resolve_context(state):
    question = state["question"]
    history = state.get("history", [])

    prompt = f"""You are a helpful assistant. Rewrite the current question with full context.

    History:
    {format_history(history)}

    Current question:
    {question}

    Rewritten (context-resolved) question:"""

    resolved_question = llm(prompt) 
    state["resolved_question"] = resolved_question.strip()
    return state


def format_history(history):
    return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

def find_related_products(query_text, top_k=5):
    try:
        # Encode text to vector using your model
        embedding_vector = model.encode(query_text)
        
        # Ensure it's a plain list of floats, not NumPy or torch tensor
        if not isinstance(embedding_vector, list):
            embedding_vector = embedding_vector.tolist()
        
        # Verify the embedding is the right type and dimension
        print(f"Embedding type: {type(embedding_vector)}")
        print(f"Embedding length: {len(embedding_vector)}")
        print(f"First few values: {embedding_vector[:3]}")
        
        # Use the correct Cypher query for vector search
        cypher_query = """
        CALL db.index.vector.queryNodes('product_embedding_index', $topK, $embedding)
        YIELD node, score
        WHERE score > 0.75
       RETURN 
          node.id AS id, 
          node.name AS name, 
          node.description AS description, 
          score
        ORDER BY score DESC
        """

        faq_query = """
        CALL db.index.vector.queryNodes('faq_embedding', $topK, $embedding)
        YIELD node, score
        WHERE score >= 0.75
        RETURN 
            'FAQ' AS type,
            node.id AS id, 
            node.question AS question,
            node.answer AS answer,
            score
        ORDER BY score DESC
        """
        
        params = {
            "embedding": (embedding_vector),
            "topK": int(top_k)
        }

        


        



        result = enhanced_graph.query(cypher_query, params=params)
        faq_results = enhanced_graph.query(faq_query, params=params)
        print("result",result)
        print("faq_results",faq_results)



        all_results = result + faq_results
        all_results_sorted = sorted(all_results, key=lambda r: r['score'], reverse=True)
        return all_results_sorted
    
        
    except Exception as e:
        
        return "Error"
def classify_question_manually(question):
    related_products = find_related_products(question)
   
    return related_products
    
    # Check if we got valid results
    
    



guardrails_system = """As an intelligent assistant, your primary objective is to decide whether a given question is related to neo4j schema or not. 
If the question is related to neo4j schema, output "related". Otherwise, output "end".
To make this decision, assess the content of the question and determine if it matches with neo4j schema, 
or related topics. Provide only the specified output: "related" or "end".
Do not genrate syntax error"""
guardrails_prompt = ChatPromptTemplate.from_messages(
    [
        ( 
            "system",
            guardrails_system,
        ),
        (
            "human",
            ("""Neo4j Schema: {enhanced_grap}
Question: {question}"""),
        ),
    ]
)
                                                          

class GuardrailsOutput(BaseModel):
    decision: Literal["related", "end"] = Field(
        description="Decision on whether the question is related to our database . To find Out That YOU need to check with that our schema"
    )

def log_output(output: GuardrailsOutput) -> GuardrailsOutput:
    
    
    return GuardrailsOutput(output)
    
    
    
guardrails_chain = guardrails_prompt | log_output | llm.with_structured_output(GuardrailsOutput)


known_products = ['coaster']  # etc.

def guardrails(state:OverallState)->OverallState:
    related_products = classify_question_manually(state["question"])
    state["related_products"] = related_products

    print(related_products,"related_products")   
    

    if related_products is None or len(related_products) == 0:
        decision= "END"  # or whatever your END constant is
    else:
        decision= "related"    
    

    return {
        "question": state["question"],
        "schema":enhanced_grap,
        "next_action": decision,
        "related_products": related_products
    }
# feed examples



examples = [
    {
        "question": "How many products are there?",
        "query": "MATCH (n:Product) RETURN count(n)",
    },
    
     {
        "question": "Which products are available in table cover?",
        "query": "MATCH (n:Category)-[:HAS_PRODUCT]->(p:Product) where toLower(n.name) contains 'table cover' RETURN p.name",
    },
    {
        "question": "What thread types are available for coasters?",
        "query": """MATCH (c:Category)-[:HAS_PRODUCT]->(p:Product)
                    WHERE toLower(p.name) CONTAINS 'coaster'
                    MATCH (p)-[:HAS_OPTION]->(o:Option)
                    WHERE toLower(o.name) CONTAINS 'thread type'
                    MATCH (o)-[:HAS_VALUE]->(v:OptionValue)
                    RETURN DISTINCT v.name AS thread_types , p.name AS Coaster_Type LIMIT 15""",
    },
    {
        "question": "What cover styles do hats have?",
        "query": """MATCH (p:Product)
                WHERE toLower(p.name) CONTAINS 'hat'
                MATCH (p)-[:HAS_OPTION]->(o:Option)
                WHERE toLower(o.name) CONTAINS 'cover style'
                MATCH (o)-[:HAS_VALUE]->(v:OptionValue)
                RETURN DISTINCT v.name AS cover_styles,p.name AS PRODUCT_Type""",
            },
    {
        "question": "List all available patch sizes for caps",
        "query": """MATCH (p:Product)
                WHERE toLower(p.name) CONTAINS 'cap'
                MATCH (p)-[:HAS_OPTION]->(o:Option)
                WHERE toLower(o.name) CONTAINS 'patch size'
                MATCH (o)-[:HAS_VALUE]->(v:OptionValue)
                RETURN DISTINCT v.name AS patch_sizes ,p.name AS PRODUCT_Type LIMIT 10""",
    },
    {
        "question": "Do candles require sample proof?",
        "query": """MATCH (p:Product)
            WHERE toLower(p.name) CONTAINS 'candle'
            MATCH (p)-[:HAS_OPTION]->(o:Option)
            WHERE toLower(o.name) CONTAINS 'sample proof required'
            MATCH (o)-[:HAS_VALUE]->(v:OptionValue)
            RETURN DISTINCT v.name AS sample_proof_options""",
    },
    {
        "question": "What mouse pad thickness options are available for covers?",
        "query": """MATCH (p:Product)
            WHERE toLower(p.name) CONTAINS 'cover'
            MATCH (p)-[:HAS_OPTION]->(o:Option)
            WHERE toLower(o.name) CONTAINS 'mouse pad thickness'
            MATCH (o)-[:HAS_VALUE]->(v:OptionValue)
            RETURN DISTINCT v.name AS thickness_options LIMIT 10""",
            },
    {
        "question": "Which table sizes are available for table covers?",
        "query": """MATCH (p:Product)
            WHERE toLower(p.name) CONTAINS 'table cover'
            MATCH (p)-[:HAS_OPTION]->(o:Option)
            WHERE toLower(o.name) CONTAINS 'table size'
            MATCH (o)-[:HAS_VALUE]->(v:OptionValue)
            RETURN DISTINCT v.name AS table_sizes LIMIT 10""",
            },
    {
        "question": "What patch backing options exist for hats?",
        "query": """MATCH (p:Product)
            WHERE toLower(p.name) CONTAINS 'hat'
            MATCH (p)-[:HAS_OPTION]->(o:Option)
            WHERE toLower(o.name) CONTAINS 'patch backing'
            MATCH (o)-[:HAS_VALUE]->(v:OptionValue)
            RETURN DISTINCT v.name AS patch_backing_options""",
            },
    {
        "question": "Is individual polybag packaging available for coasters?",
        "query": """MATCH (p:Product)
            WHERE toLower(p.name) CONTAINS 'coaster'   
            MATCH (p)-[:HAS_OPTION]->(o:Option)
            WHERE toLower(o.name) CONTAINS 'individual polybag'
            MATCH (o)-[:HAS_VALUE]->(v:OptionValue)
            RETURN DISTINCT v.name AS polybag_options """,
            },
    {
        "question": "List all table runner sizes for table covers",
        "query": """MATCH (p:Product)
            WHERE toLower(p.name) CONTAINS 'table cover'
            MATCH (p)-[:HAS_OPTION]->(o:Option)
            WHERE toLower(o.name) CONTAINS 'table runner size'
            MATCH (o)-[:HAS_VALUE]->(v:OptionValue)
            RETURN DISTINCT v.name AS runner_sizes LIMIT 10""",
    },
    {
        "question": "Show me mouse pad material options for covers",
        "query": """MATCH (p:Product)
            WHERE toLower(p.name) CONTAINS 'cover'
            MATCH (p)-[:HAS_OPTION]->(o:Option)
            WHERE toLower(o.name) CONTAINS 'mouse pad material'
            MATCH (o)-[:HAS_VALUE]->(v:OptionValue)
            RETURN DISTINCT v.name AS material_options LIMIT 10""",
    },
    {
        "question": "What drop height values are there for table covers?",
        "query": """MATCH (p:Product)
            WHERE toLower(p.name) CONTAINS 'table cover'
            MATCH (p)-[:HAS_OPTION]->(o:Option)
            WHERE toLower(o.name) CONTAINS 'drop height'
            MATCH (o)-[:HAS_VALUE]->(v:OptionValue)
            RETURN DISTINCT v.name AS drop_heights""",
    },
    {
        "question": "What options exist for back side of candles?",
        "query": """MATCH (p:Product)
            WHERE toLower(p.name) CONTAINS 'candle'
            MATCH (p)-[:HAS_OPTION]->(o:Option)
            WHERE toLower(o.name) CONTAINS 'back side'
            MATCH (o)-[:HAS_VALUE]->(v:OptionValue)
            RETURN DISTINCT v.name AS back_side_options""",
    },
    {
        "question":"Ccoaster are available in 3 inch",
        "query": """ MATCH (c:Category)-[:HAS_PRODUCT]->(p:Product)
                WHERE toLower(c.name) CONTAINS 'coaster'
                MATCH (p)-[:HAS_OPTION]->(o:Option)
                WHERE toLower(o.name) CONTAINS 'size'
                MATCH (o)-[:HAS_VALUE]->(v:OptionValue)
                WHERE v.name CONTAINS '3'
                RETURN DISTINCT c.name AS coaster_name ,  v.name AS SIZE  LIMIT 10""",
    },

    {
        "question":"What is a table skirt?",
        "query":"""MATCH (c:Category)-[:HAS_PRODUCT]->(p:Product)
WHERE toLower(p.name) CONTAINS 'table skirt'
RETURN p.name AS table_skirt_product
"""
    },
  
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, OpenAIEmbeddings(), Neo4jVector, k=10, input_keys=["question"]
)



text2cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Given an input question, convert it to a Cypher query. No pre-amble."
                "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
                "Also the serch for sementic serch "
                "Example:'If user is ask for Birthday party than find reletable for birthday party and create cypher for that'"
            ),
        ),
        (
            "human",
            (
                """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.
                    Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!
                    Here is the schema information
{schema}

Below are a number of examples of questions and their corresponding Cypher queries.

{fewshot_examples}

User input: {question}
Cypher query:"""
            ),  
        ),
    ]
)

text2cypher_chain = text2cypher_prompt | llm | StrOutputParser()


def generate_cypher(state: OverallState) -> OverallState:
    """
    Generates a cypher statement based on the provided schema and user input
    """
    NL = "___"
    fewshot_examples = (NL * 2).join(
        [
            f"Question: {el['question']}{NL}Cypher:{el['query']}"
            for el in example_selector.select_examples(
                {"question": state.get("question")}
            )
        ]
    )
    
  
    generated_cypher = text2cypher_chain.invoke(
        {
            "question": state.get("question"),
            "fewshot_examples": fewshot_examples,
            "schema": enhanced_graph,
        }
    )
    return {"cypher_statement": generated_cypher, "steps": ["generate_cypher"]}


from typing import List, Optional

validate_cypher_system = """
You are a Cypher expert reviewing a statement written by a user .
"""

validate_cypher_user = """You must check the following:
* Are there any syntax errors in the Cypher statement?
* Are there any missing or undefined variables in the Cypher statement?
* Are any node labels missing from the schema?
* Are any relahip tionstypes missing from the schema?
* Are any of the properties not included in the schema?
* Does the Cypher statement include enough information to answer the question?

Examples of good errors:
* Label (:Foo) does not exist, did you mean (:Bar)?
* Property bar does not exist for label Foo, did you mean baz?
* Relationship FOO does not exist, did you mean FOO_BAR?

Schema:
{schema}

The question is:
{question}

The Cypher statement is:
{cypher}

Make sure you don't make any mistakes!"""

validate_cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            validate_cypher_system,
        ),
        (
            "human",
            (validate_cypher_user),
        ),
    ]
)


class Property(BaseModel):
    """
    Represents a filter condition based on a specific node property in a graph in a Cypher statement.
    """

    node_label: str = Field(
        description="The label of the node to which this property belongs."
    )
    property_key: str = Field(description="The key of the property being filtered.")
    property_value: str = Field(
        description="The value that the property is being matched against."
    )

   
class ValidateCypherOutput(BaseModel):
    """
    Represents the validation result of a Cypher query's output,
    including any errors and applied filters.
    """

    errors: Optional[List[str]] = Field(
        description="A list of syntax or semantical errors in the Cypher statement. Always explain the discrepancy between schema and Cypher statement"
    )
    filters: Optional[List[Property]] = Field(
        description="A list of property-based filters applied in the Cypher statement."
    )

#for the validate the cypher
validate_cypher_chain = validate_cypher_prompt | llm.with_structured_output(
    ValidateCypherOutput
)


# Cypher query corrector is experimental
corrector_schema = [
    Schema(el["start"], el["type"], el["end"])
    for el in enhanced_graph.structured_schema.get("relationships")
]
cypher_query_corrector = CypherQueryCorrector(corrector_schema)


def validate_cypher(state: OverallState) -> OverallState:
    """
    Validates the Cypher statements and maps any property values to the database.
    """
    errors = []
    mapping_errors = []
    # Check for syntax errors
    try:
        enhanced_graph.query(f"EXPLAIN {state.get('cypher_statement')}")
    except CypherSyntaxError as e:
        errors.append(e.message)
    # Experimental feature for correcting relationship directions
    corrected_cypher = cypher_query_corrector(state.get("cypher_statement"))
    if not corrected_cypher:
        errors.append("The generated Cypher statement doesn't fit the graph schema")
    if not corrected_cypher == state.get("cypher_statement"):
        print("Relationship direction was corrected")
    # Use LLM to find additional potential errors and get the mapping for values
    llm_output = validate_cypher_chain.invoke(
        {
            "question": state.get("question"),
            "schema": enhanced_graph,
            "cypher": state.get("cypher_statement"),
        }
    )
    if llm_output.errors:
        errors.extend(llm_output.errors)
    if llm_output.filters:
        for filter in llm_output.filters:
            # Do mapping only for string values
            if (
                not [
                    prop
                    for prop in enhanced_graph.structured_schema["node_props"][
                        filter.node_label
                    ]
                    if prop["property"] == filter.property_key
                ][0]["type"]
                == "STRING"
            ):
                continue
            mapping = enhanced_graph.query(
                f"MATCH (n:{filter.node_label}) WHERE toLower(n.`{filter.property_key}`) = toLower($value) RETURN 'yes' LIMIT 1",
                {"value": filter.property_value},
            )
            if not mapping:
                print(
                    f"Missing value mapping for {filter.node_label} on property {filter.property_key} with value {filter.property_value}"
                )
                mapping_errors.append(
                    f"Missing value mapping for {filter.node_label} on property {filter.property_key} with value {filter.property_value}"
                )
    if mapping_errors:
        next_action = "end"
    elif errors:
        next_action = "correct_cypher"
    else:
        next_action = "execute_cypher"

    return {
        "next_action": next_action,
        "cypher_statement": corrected_cypher,
        "cypher_errors": errors,
        "steps": ["validate_cypher"],
    }


correct_cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a Cypher expert reviewing a statement written by a junior developer. "
                "You need to correct the Cypher statement based on the provided errors. No pre-amble."
                "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
            ),
        ),
        (
            "human",
            (
                """Check for invalid syntax or semantics and return a corrected Cypher statement.

Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not wrap the response in any backticks or anything else.
Respond with a Cypher statement only!

Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.

The question is:
{question}

The Cypher statement is:
{cypher}

The errors are:
{errors}

Corrected Cypher statement: """
            ),
        ),
    ]
)

correct_cypher_chain = correct_cypher_prompt | llm | StrOutputParser()

print("printing correct_cypher_chain",correct_cypher_chain)


def correct_cypher(state: OverallState) -> OverallState:
    """
    Correct the Cypher statement based on the provided errors.
    """
    print("Inside The  correct_cypher")
    corrected_cypher = correct_cypher_chain.invoke(
        {
            "question": state.get("question"),
            "errors": state.get("cypher_errors"),
            "cypher": state.get("cypher_statement"),
            "schema": enhanced_grap,
        }
    )

    return {
        "next_action": "validate_cypher",
        "cypher_statement": corrected_cypher,
        "steps": ["correct_cypher"],
    }


no_results = "Can you explin this in lettle bit in detail"


def execute_cypher(state: OverallState) -> OverallState:
    """
    Executes the given Cypher statement.
    """
   
    related_products =state.get("related_products")
   
    records = enhanced_graph.query(state.get("cypher_statement"))
    print("Test For Neo4j")
    print(records,state.get("cypher_statement"))
    if len(records) ==0:
        records =related_products
    

    return {
        "database_records": records ,
        "next_action": "end",
        "steps": ["execute_cypher"],
    }


generate_final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant ",
        ),
        (
            "human",
            (
                """Use the following results retrieved from a database to provide
a succinct, definitive answer to the user's question. iF you do not get propare  ans than ask to use in little bit in depth .You are Ansring the customes of e commers web site to give ans in that way.

Respond as if you are answering the question directly.

Results: {results}

Question: {question}"""
            ),
        ),
    ]
)

generate_final_chain = generate_final_prompt | llm | StrOutputParser()


def generate_final_answer(state: OverallState) -> OutputState:
    """
    Decides if the question is related to Neo4j database.
    """
    database_records=state.get("database_records")
    related_products=state.get("related_products")

    print(related_products)

    if database_records is None:
        database_records =related_products

   
    final_answer = generate_final_chain.invoke(
        {"question": state.get("question"), "results": database_records}
    )
    
    history = state.get("history", [])
    history.append({"role": "user", "content": state["question"]})
    history.append({"role": "assistant", "content": final_answer})
    state["history"] = history
    print("Printing History ",history)
    return {"answer": final_answer, "steps": ["generate_final_answer"]}

def guardrails_condition(
    state: OverallState,
) -> Literal["generate_cypher", "generate_final_answer"]:
    if state.get("next_action") == "end":
        return "generate_final_answer"
    elif state.get("next_action") == "related":
        return "generate_cypher"
    


def validate_cypher_condition(
    state: OverallState,
) -> Literal["generate_final_answer", "correct_cypher", "execute_cypher"]:
    if state.get("next_action") == "correct_cypher":
        return "execute_cypher"
    
    elif state.get("next_action") == "correct_cypher":
        return "execute_cypher"  
    elif state.get("next_action") == "execute_cypher":
        return "generate_final_answer"
    
    return "execute_cypher"



langgraph = StateGraph(OverallState, input=InputState, output=OutputState)
langgraph.add_node(guardrails)
langgraph.add_node(generate_cypher)
langgraph.add_node(validate_cypher)
langgraph.add_node(correct_cypher)
langgraph.add_node(execute_cypher)
langgraph.add_node(generate_final_answer)

langgraph.add_edge(START, "guardrails")
langgraph.add_conditional_edges(
    "guardrails",
    guardrails_condition,
)
langgraph.add_edge("generate_cypher", "validate_cypher")
langgraph.add_conditional_edges(
    "validate_cypher",
    validate_cypher_condition,
)
langgraph.add_edge("execute_cypher", "generate_final_answer")
langgraph.add_edge("correct_cypher", "validate_cypher")
langgraph.add_edge("generate_final_answer", END)

langgraph = langgraph.compile()


print(langgraph.get_graph().draw_mermaid())

response = langgraph.invoke({"question": "What other types/ kinds of bottle openers do you offer? "})

print(response)    
