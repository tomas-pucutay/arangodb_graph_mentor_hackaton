from src.utils.base_schema import base_arquitecture
from src.ingestion.helpers import get_edge_type
from src.ingestion.generators.chains import knowledge_chain
from src.ingestion.generators.nodes import *

import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from tqdm import tqdm

# Knowledge generator

# KNOWLEDGE

# Factors
# 27 competences
# 58 specialization areas
# 13 industries

# Stats
# Tokens usage: 12,516,055
# Cost: $8.22

knowledge_edges = []
knowledge_nodes = []

def knowledge_combination(c, s, i):
    prefix = f"{c['id']}_{s['id']}_{i['id']}_"

    response = knowledge_chain.invoke({
        "competence": c['node_name'],
        "specialization_area": s['node_name'],
        "industry": i['node_name']
    })

    edges = []
    nodes = []

    for seq, element in enumerate(response.node_detail):
        knowledge_suffix = f"KNW_{seq+1}"

        nodes.append({
            'id': prefix + knowledge_suffix,
            'node_name': element.node_name,
            'node_category': response.node_type
        })

        # Edges from factors to knowledges
        for val in [c, s, i]:
            edges.append({
                "source": val['id'],
                "target": prefix + knowledge_suffix,
                "edge_type": get_edge_type(val['node_category'], response.node_type, base_arquitecture)
            })

        for seq, action_plan in enumerate(element.action_plan):
            action_suffix = f"ACT_{seq+1}"

            nodes.append({
                'id': prefix + knowledge_suffix + '_' + action_suffix,
                'node_name': action_plan.name,
                'node_category': 'action_plan',
                'duration': action_plan.duration,
            })

            # Edges from knowledges to action_plans
            edges.append({
                "source": prefix + knowledge_suffix,
                "target": prefix + knowledge_suffix + '_' + action_suffix,
                "edge_type": get_edge_type(response.node_type, 'action_plan', base_arquitecture),
            })

            # Edges from action_plans to competences
            edges.append({
                "source": prefix + knowledge_suffix + '_' + action_suffix,
                "target": c['id'],
                "edge_type": get_edge_type('action_plan', c['node_category'], base_arquitecture),
                "weight": action_plan.weight
            })

    return nodes, edges

combinations = list(product(competence_nodes, specialization_area_nodes, industry_nodes))

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(knowledge_combination, c, s, i): (c, s, i) for c, s, i in combinations}

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="task"):
        nodes, edges = future.result()
        knowledge_nodes.extend(nodes)
        knowledge_edges.extend(edges)

joblib.dump(knowledge_nodes, "knowledge_nodes.pkl", compress=3)
joblib.dump(knowledge_edges, "knowledge_edges.pkl", compress=3)