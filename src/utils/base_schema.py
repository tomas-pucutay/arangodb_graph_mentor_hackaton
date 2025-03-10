import matplotlib.pyplot as plt
import networkx as nx

base_arquitecture = {
    "main_node": ["competence"],
    "nodes": [
        "skill",
        "knowledge",
        "attitude",
        "profession",
        "specialization_area",
        "industry",
        "experience_level",
        "responsibility_level",
        "organizational_culture",
        "action_plan"
        ],
    "edges": [
        # From competence to the 3 main factors: skill, knowledge and attitude
        {"source": "competence", "target": "skill", "edge_type": "contains"},
        {"source": "competence", "target": "knowledge", "edge_type": "requires"},
        {"source": "competence", "target": "attitude", "edge_type": "needs"},
        # Factors that cause variation in skill
        {"source": "responsibility_level", "target": "skill", "edge_type": "determines"},
        {"source": "profession", "target": "skill", "edge_type": "relevant_for"},
        # Factors that cause variation in knowledge
        {"source": "specialization_area", "target": "knowledge", "edge_type": "influences"},
        {"source": "industry", "target": "knowledge", "edge_type": "relevant_for"},
        # Factors that cause variation in attitude
        {"source": "organizational_culture", "target": "attitude", "edge_type": "influences"},
        {"source": "experience_level", "target": "attitude", "edge_type": "shapes"},
        # Concrete actions from skill, knowledge and attitude
        {"source": "knowledge", "target": "action_plan", "edge_type": "guides"},
        {"source": "attitude", "target": "action_plan", "edge_type": "guides"},
        {"source": "skill", "target": "action_plan", "edge_type": "reinforces"},
        # Final result
        {"source": "action_plan", "target": "competence", "edge_type": "develops"}
    ]
}

# Visualization

G = nx.DiGraph()

for node in base_arquitecture["main_node"] + base_arquitecture["nodes"]:
    G.add_node(node)

edges_with_labels = {}
for edge in base_arquitecture["edges"]:
    G.add_edge(edge["source"], edge["target"])
    edges_with_labels[(edge["source"], edge["target"])] = edge["edge_type"]

plt.figure(figsize=(7, 7))

pos = nx.spring_layout(G)
pos["competence"] = (0, 0)
pos["skill"] = (-0.25, 0.25)
pos["knowledge"] = (0.25, 0.25)
pos["attitude"] = (0, -0.25)

node_colors = [
    "#FFCCCB" if node == "competence" else
    "#FFD580" if node in ["skill", "knowledge", "attitude"] else
    "#A7C7E7" if node == "action_plan" else
    "lightgray" for node in G.nodes
    ]

nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color="gray",
        node_size=2000, font_size=8, font_weight="bold", arrows=True, alpha=0.8)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edges_with_labels, font_size=8)

plt.title("Knowledge Graph Visualization", fontsize=14)
plt.show()