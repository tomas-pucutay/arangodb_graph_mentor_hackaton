from src.utils.base_schema import base_arquitecture
from src.ingestion.helpers import get_edge_type
from src.ingestion.generators.chains import skill_chain
from src.ingestion.generators.nodes import *

import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from tqdm import tqdm

# Skills generator

# SKILLS

# Factors
# 27 competences
# 18 profession
# 3 responsibility levels

# Stats
# Tokens usage: 988,377
# Cost: $0.64

skill_edges = []
skill_nodes = []

def skill_combination(c, p, r):
    prefix = f"{c['id']}_{p['id']}_{r['id']}_"

    response = skill_chain.invoke({
        "competence": c['node_name'],
        "profession": p['node_name'],
        "responsibility_level": r['node_name']
    })

    edges = []
    nodes = []

    for seq, element in enumerate(response.node_detail):
        skill_suffix = f"SKL_{seq+1}"

        nodes.append({
            'id': prefix + skill_suffix,
            'node_name': element.node_name,
            'node_category': response.node_type
        })

        # Edges from factors to skills
        for val in [c, p, r]:
            edges.append({
                "source": val['id'],
                "target": prefix + skill_suffix,
                "edge_type": get_edge_type(val['node_category'], response.node_type, base_arquitecture)
            })

        for seq, action_plan in enumerate(element.action_plan):
            action_suffix = f"ACT_{seq+1}"

            nodes.append({
                'id': prefix + skill_suffix + '_' + action_suffix,
                'node_name': action_plan.name,
                'node_category': 'action_plan',
                'duration': action_plan.duration,
            })

            # Edges from skills to action_plans
            edges.append({
                "source": prefix + skill_suffix,
                "target": prefix + skill_suffix + '_' + action_suffix,
                "edge_type": get_edge_type(response.node_type, 'action_plan', base_arquitecture),
            })

            # Edges from action_plans to competences
            edges.append({
                "source": prefix + skill_suffix + '_' + action_suffix,
                "target": c['id'],
                "edge_type": get_edge_type('action_plan', c['node_category'], base_arquitecture),
                "weight": action_plan.weight
            })

    return nodes, edges

combinations = list(product(competence_nodes, profession_nodes, responsibility_level_nodes))

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(skill_combination, c, p, r): (c, p, r) for c, p, r in combinations}

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="task"):
        nodes, edges = future.result()
        skill_nodes.extend(nodes)
        skill_edges.extend(edges)

joblib.dump(skill_nodes, "skill_nodes.pkl", compress=3)
joblib.dump(skill_edges, "skill_edges.pkl", compress=3)