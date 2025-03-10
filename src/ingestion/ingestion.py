from src.ingestion.data.cleaning import competence_nodes, factor_nodes, driver_nodes, action_plan_nodes
from src.ingestion.data.cleaning import competence_to_driver_edges, factor_to_driver_edges
from src.ingestion.data.cleaning import driver_to_action_plan_edges, action_plan_to_competence_edges
from src.utils.enums import *
import os

from dotenv import load_dotenv
from arango import ArangoClient

load_dotenv()

# Const
PASSWORD = os.getenv('ARANGODB_PASSWORD')
USER = os.getenv('ARANGODB_USER')
HOSTNAME = os.getenv('ARANGODB_HOSTNAME')

# Upload data into ArangoDB

client = ArangoClient(hosts=HOSTNAME)
sys_db = client.db("_system", username=USER, password=PASSWORD)
if not sys_db.has_database("work_db"):
    sys_db.create_database("work_db")
work_db = client.db("work_db", username=USER, password=PASSWORD)

GRAPH_NAME = "graph_mentor"
COLLECTIONS = {
    "nodes": [
        "Competence",
        "Factor",
        "Driver",
        "Action_Plan"
        ],
    "edges": [
        "Competence_to_Driver",
        "Factor_to_Driver",
        "Driver_to_Action_Plan",
        "Action_Plan_to_Competence"]
}

def create_collections(db):

    # For nodes
    for collection in COLLECTIONS["nodes"]:
        if not db.has_collection(collection):
            db.create_collection(collection)
            print(f"Created collection: {collection}")
            
            db.collection(collection).add_index({'type': 'hash', 'fields': ['node_category'], 'unique': False})
            db.collection(collection).add_index({'type': 'fulltext', 'fields': ['node_name']})

            if collection == "Action_Plan":
                db.collection(collection).add_index({'type': 'persistent', 'fields': ['duration'], 'sparse': False})

        else:
            print(f"Collection already exists: {collection}")
    
    # For edges
    for edge_collection in COLLECTIONS["edges"]:
        if not db.has_collection(edge_collection):
            db.create_collection(edge_collection, edge=True)
            print(f"Created edge collection: {edge_collection}")

            db.collection(edge_collection).add_index({'type': 'hash', 'fields': ['edge_type'], 'unique': False})

            if edge_collection == "Action_Plan_to_Competence":
                db.collection(edge_collection).add_index({'type': 'persistent', 'fields': ['weight'], 'sparse': False})

        else:
            print(f"Edge collection already exists: {edge_collection}")

def create_graph(db):
    if not db.has_graph(GRAPH_NAME):
        graph = db.create_graph(GRAPH_NAME)

        graph.create_edge_definition(
            edge_collection="Competence_to_Driver",
            from_vertex_collections=["Competence"],
            to_vertex_collections=["Driver"]
        )
        
        graph.create_edge_definition(
            edge_collection="Factor_to_Driver",
            from_vertex_collections=["Factor"],
            to_vertex_collections=["Driver"]
        )
        
        graph.create_edge_definition(
            edge_collection="Driver_to_Action_Plan",
            from_vertex_collections=["Driver"],
            to_vertex_collections=["Action_Plan"]
        )
        
        graph.create_edge_definition(
            edge_collection="Action_Plan_to_Competence",
            from_vertex_collections=["Action_Plan"],
            to_vertex_collections=["Competence"]
        )
        
        print(f"Created graph: {GRAPH_NAME}")
    else:
        print(f"Graph already exists: {GRAPH_NAME}")

create_collections(work_db)
create_graph(work_db)

# Upload data to ArangoDB

def insert_documents(db, collection_name, documents, batch_size=5000):
    if not documents:
        print(f"No documents to insert into {collection_name}")
        return
        
    collection = db.collection(collection_name)
    
    successful_inserts = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        try:
            result = collection.import_bulk(batch, on_duplicate="update")
            successful_inserts += result["created"] + result["updated"]
        except Exception as e:
            print(f"Error inserting batch into {collection_name}: {e}")
    
    print(f"Successfully inserted/updated {successful_inserts} documents in {collection_name}")

# Add nodes
insert_documents(work_db, "Competence", competence_nodes)
insert_documents(work_db, "Factor", factor_nodes)
insert_documents(work_db, "Driver", driver_nodes)
insert_documents(work_db, "Action_Plan", action_plan_nodes)

# Add edges
insert_documents(work_db, "Competence_to_Driver", competence_to_driver_edges)
insert_documents(work_db, "Factor_to_Driver", factor_to_driver_edges)
insert_documents(work_db, "Driver_to_Action_Plan", driver_to_action_plan_edges)
insert_documents(work_db, "Action_Plan_to_Competence", action_plan_to_competence_edges)

# Create views

# View action_plan_competence_view

work_db.create_arangosearch_view(
    name='action_plan_competence_view',
    properties={
        "links": {
            "Action_Plan": {
                "analyzers": ["text_en"],
                "fields": {
                    "node_name": { "analyzers": ["text_en"] },
                    "duration": {}
                },
                "includeAllFields": False
            },
            "Action_Plan_to_Competence": {
                "fields": {
                    "weight": {}
                }
            },
            "Competence": {
                "fields": {
                    "node_name": { "analyzers": ["text_en"] }
                }
            }
        },
        "primarySort": [
            {"field": "weight", "direction": "desc"}
        ],
        "storedValues": [
            ["weight", "node_name"]
        ]
    }
)