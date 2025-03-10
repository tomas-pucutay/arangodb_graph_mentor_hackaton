from src.utils.base_schema import base_arquitecture
from src.ingestion.helpers import get_edge_type
from src.ingestion.generators.chains import attitude_chain
from src.ingestion.generators.nodes import *

import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from tqdm import tqdm

# Attitude generator

# ATTITUDE

# Factors
# 27 competences
# 18 organizational culture
# 4 experience levels

# Stats
# Tokens usage: 1,320,818
# Cost: $0.88

attitude_edges = []
attitude_nodes = []

def attitude_combination(c, o, e):
    prefix = f"{c['id']}_{o['id']}_{e['id']}_"

    response = attitude_chain.invoke({
        "competence": c['node_name'],
        "organizational_culture": o['node_name'],
        "experience_level": e['node_name']
    })

    edges = []
    nodes = []

    for seq, element in enumerate(response.node_detail):
        attitude_suffix = f"ATT_{seq+1}"

        nodes.append({
            'id': prefix + attitude_suffix,
            'node_name': element.node_name,
            'node_category': response.node_type
        })

        # Edges from factors to attitudes
        for val in [c, o, e]:
            edges.append({
                "source": val['id'],
                "target": prefix + attitude_suffix,
                "edge_type": get_edge_type(val['node_category'], response.node_type, base_arquitecture)
            })

        for seq, action_plan in enumerate(element.action_plan):
            action_suffix = f"ACT_{seq+1}"

            nodes.append({
                'id': prefix + attitude_suffix + '_' + action_suffix,
                'node_name': action_plan.name,
                'node_category': 'action_plan',
                'duration': action_plan.duration,
            })

            # Edges from attitudes to action_plans
            edges.append({
                "source": prefix + attitude_suffix,
                "target": prefix + attitude_suffix + '_' + action_suffix,
                "edge_type": get_edge_type(response.node_type, 'action_plan', base_arquitecture),
            })

            # Edges from action_plans to competences
            edges.append({
                "source": prefix + attitude_suffix + '_' + action_suffix,
                "target": c['id'],
                "edge_type": get_edge_type('action_plan', c['node_category'], base_arquitecture),
                "weight": action_plan.weight
            })

    return nodes, edges

combinations = list(product(competence_nodes, organizational_culture_nodes, experience_level_nodes))

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(attitude_combination, c, o, e): (c, o, e) for c, o, e in combinations}

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="task"):
        nodes, edges = future.result()
        attitude_nodes.extend(nodes)
        attitude_edges.extend(edges)

joblib.dump(attitude_nodes, "attitude_nodes.pkl", compress=3)
joblib.dump(attitude_edges, "attitude_edges.pkl", compress=3)