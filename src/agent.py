# Standard modules
from src.utils.enums import Competence
from src.utils.helpers import get_competences
import os

# Third-party libraries
import nx_arangodb as nxadb
from arango import ArangoClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Langchain and Langgraph
from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain_community.graphs import ArangoGraph
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Const
PASSWORD = os.getenv('ARANGODB_PASSWORD')
USER = os.getenv('ARANGODB_USER')
HOSTNAME = os.getenv('ARANGODB_HOSTNAME')

llm = ChatOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    model="gpt-4o-mini",
    temperature=0
)

client = ArangoClient(hosts=HOSTNAME)
sys_db = client.db("_system", username=USER, password=PASSWORD)
if not sys_db.has_database("work_db"):
    sys_db.create_database("work_db")
work_db = client.db("work_db", username=USER, password=PASSWORD)


# Create tools

@tool
def find_competences() -> dict:
    """
    This tool extracts competences when the user ask for them.
    """
    return get_competences(work_db)

def find_closest_competence(llm, query):
    """
    This tool helps the user to find a suitable competence. They will present you an scenario
    and you need to give them a guide with the most similar competence to their case.
    """
    
    class Selection(BaseModel):
        competence: Competence = Field(
            description="Given a text, clasify the closest competence",
        )

    system = """You are a classification system, given a user query assign the most similar competence"""
    structured_llm = llm.with_structured_output(Selection)

    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "User query: {query}")]
    )

    chain = prompt | structured_llm
    response = chain.invoke({'query':query})
    return response.competence.value

@tool
def find_suitable_competence(llm, query):
    """
    This tool helps the user to find a suitable competence. They will present you an scenario
    and you need to give them a guide with the most similar competence to their case.
    """
    
    return find_closest_competence(llm, query)

@tool
def find_best_competences_for_action_plan(query: str):
    """
    This tool identifies the best competences being developed by a given action plan.
    It tokenizes the action plan text, searches for similar matching plans,
    joins them with their associated competences and returns the top 3 competences
    based on the average weight of the top 5 associated plans.
    """

    aql = """
    LET keywords = TOKENS(@user_text, 'text_en')

    // Filter candidates
    LET candidatePlans = (
      FOR plan IN action_plan_competence_view
        SEARCH ANALYZER(
          plan.node_name IN keywords,
          'text_en'
        )
        SORT BM25(plan) DESC
        LIMIT 20000
        RETURN plan
    )

    // Join candidates and competences
    LET candidateEdges = (
      FOR plan IN candidatePlans
        LET edge = FIRST(
          FOR e IN Action_Plan_to_Competence
            FILTER e._from == plan._id
            RETURN e
        )
        FILTER edge != null
        RETURN { plan, edge }
    )

    // Join competence to find node_name
    LET competenceCandidates = (
      FOR pair IN candidateEdges
        LET comp = DOCUMENT(pair.edge._to)
        RETURN {
          competence_id: comp._id,
          competence_name: comp.node_name,
          plan_id: pair.plan._id,
          plan_name: pair.plan.node_name,
          weight: pair.edge.weight
        }
    )

    // Groupby and get average weight, select top 5 plans per competence
    LET groupedByCompetence = (
      FOR cand IN competenceCandidates
        COLLECT comp_id = cand.competence_id, comp_name = cand.competence_name INTO plansArr = cand
        LET sortedPlans = (
          FOR p IN plansArr
            SORT p.weight DESC
            RETURN { plan_name: p.plan_name, weight: p.weight }
        )
        LET top5 = SLICE(sortedPlans, 0, 5)
        LET avg_weight = LENGTH(top5) > 0 ? AVERAGE(top5[*].weight) : 0
        RETURN {
          competence_name: comp_name,
          avg_weight,
          plans: top5
        }
    )

    // Rank and give top 3 developed competences
    LET ranked_competences = (
      FOR comp IN groupedByCompetence
        SORT comp.avg_weight DESC
        LIMIT 3
        RETURN comp
    )

    RETURN ranked_competences
    """

    cursor = work_db.aql.execute(aql, bind_vars={"user_text": query})
    response = list(cursor)

    return response
  
@tool
def suggest_action_plans_for_competence(query: str):
    """
    This tool suggests targeted action plans to improve a specific competence.
    When a user indicates a desire to enhance a competence or that they want to improve something very vaguely
    """
    
    select_competence = find_closest_competence(llm, query)

    aql = """
    LET targetCompetence = @competence
    LET keywords = TOKENS(targetCompetence, 'text_en')

    // Get ID of the Competence
    LET targetCompetenceKey = FIRST(
      FOR comp IN Competence
        FILTER comp.node_name == targetCompetence
        RETURN comp._key
    )

    // Find possible action plans for the competence
    LET candidatePlans = (
      FOR plan IN action_plan_competence_view
        SEARCH ANALYZER(
          plan.node_name IN keywords,
          'text_en'
        )
        FILTER plan.node_category == 'action_plan'
          AND STARTS_WITH(plan._key, targetCompetenceKey)
        SORT BM25(plan) DESC
        LIMIT 20000
        RETURN plan
    )

    LET relevantAPs = (
      FOR plan IN candidatePlans
        FOR ape IN Action_Plan_to_Competence
          FILTER ape._from == plan._id
          RETURN MERGE(plan, { weight: ape.weight })
    )

    // Find 3 for each of the drivers
    LET driverAPs = UNION(
      // 3 skills
      (
        FOR ap IN relevantAPs
          FOR da IN Driver_to_Action_Plan
            FILTER da._to == ap._id
            LET driver = DOCUMENT(da._from)
            FILTER driver.node_category == "skill"
            SORT RAND()
            LIMIT 3
            RETURN { driver, ap }
      ),
      // 3 knowledges
      (
        FOR ap IN relevantAPs
          FOR da IN Driver_to_Action_Plan
            FILTER da._to == ap._id
            LET driver = DOCUMENT(da._from)
            FILTER driver.node_category == "knowledge"
            SORT RAND()
            LIMIT 3
            RETURN { driver, ap }
      ),
      // 3 attitudes
      (
        FOR ap IN relevantAPs
          FOR da IN Driver_to_Action_Plan
            FILTER da._to == ap._id
            LET driver = DOCUMENT(da._from)
            FILTER driver.node_category == "attitude"
            SORT RAND()
            LIMIT 3
            RETURN { driver, ap }
      )
    )

    // Paso 5: Agrupar por driver y seleccionar los 2 planes con mayor weight
    LET groupedDrivers = (
      FOR item IN driverAPs
        COLLECT driver = item.driver INTO apGroup = item
        LET plans = apGroup[*].ap
        LET sortedPlans = (
          FOR plan IN plans
            SORT plan.weight DESC
            RETURN { plan_name: plan.node_name, weight: plan.weight }
        )
        LET top3Plans = SLICE(sortedPlans, 0, 3)
        RETURN { driver, action_plans: top3Plans }
    )

    RETURN groupedDrivers
    """

    cursor = work_db.aql.execute(aql, bind_vars={"competence": select_competence})
    response = list(cursor)
    return response

@tool
def best_action_plan_per_competence(query:str):
    """
    This tool is used when the user ask what is the best course of action or action plan or activity they can do
    in order to improve an specific competence
    """
    
    select_competence = find_closest_competence(llm, query)

    aql = """
    LET targetCompetence = @competence
    LET keywords = TOKENS(targetCompetence, 'text_en')
    LET targetCompetenceKey = FIRST(
      FOR comp IN Competence
        FILTER comp.node_name == targetCompetence
        RETURN comp._key
    )

    LET candidatePlans = (
      FOR plan IN action_plan_competence_view
      SEARCH ANALYZER(plan.node_name IN keywords, 'text_en')
      FILTER plan.node_category == 'action_plan'
        AND STARTS_WITH(plan._key, targetCompetenceKey)
      RETURN plan
    )

    // Join action plan and weight
    LET candidateEdges = (
      FOR plan IN candidatePlans
        LET edge = FIRST(
          FOR e IN Action_Plan_to_Competence
            FILTER e._from == plan._id
            RETURN e
        )
        FILTER edge != null
        RETURN { plan, weight: edge.weight }
    )

    // Select best action plan
    LET topPlan = (
      FOR cand IN candidateEdges
        SORT cand.weight DESC
        LIMIT 1
        RETURN {
          plan_name: cand.plan.node_name,
          weight: cand.weight
        }
    )

    RETURN topPlan
    """

    cursor = work_db.aql.execute(aql, bind_vars={"competence": select_competence})
    response = list(cursor)
    return response

@tool
def find_duration_action_plan(query: str):
    """
    This tool helps the user when they want to know how muchs days a
    course of action or action plan or activity can take
    """

    aql = """
    LET keywords = TOKENS(@user_text, 'text_en')

    LET candidatePlans = (
      FOR plan IN action_plan_competence_view
        SEARCH ANALYZER(
          plan.node_name IN keywords,
          'text_en'
        )
        SORT BM25(plan) DESC
        LIMIT 20
        RETURN plan
    )

    LET durations = (
      FOR plan IN candidatePlans
        RETURN plan.duration
    )

    RETURN { average_duration: AVERAGE(durations) }
    """

    cursor = work_db.aql.execute(aql, bind_vars={"user_text": query})
    response = list(cursor)
    response[0]['average_duration'] = f"{response[0]['average_duration']} days"
    return response

G_growth = nxadb.DiGraph(
    name="graph_mentor",
    db=work_db
    )

arango_graph = ArangoGraph(work_db)

@tool
def text_to_aql_to_text(query: str):
    """This tool is available to invoke the
    ArangoGraphQAChain object, which enables you to
    translate a Natural Language Query into AQL, execute
    the query, and translate the result back into Natural Language.
    """

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

    chain = ArangoGraphQAChain.from_llm(
    	llm=llm,
    	graph=arango_graph,
    	verbose=False,
        allow_dangerous_requests=True
    )
    
    result = chain.invoke(query)
    return str(result["result"])

# Create the Agent
def create_graph_rag_agent():
    """Create and return the GraphRAG agent"""
    tools = [
        find_competences,
        find_best_competences_for_action_plan,
        suggest_action_plans_for_competence,
        best_action_plan_per_competence,
        find_duration_action_plan,
        text_to_aql_to_text,
        find_suitable_competence
    ]
    
    return create_react_agent(llm, tools)

# Function to query the agent
def query_graph_rag(query):
    """Query the GraphRAG agent with a user question"""
    agent = create_graph_rag_agent()
    final_state = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return final_state["messages"][-1].content