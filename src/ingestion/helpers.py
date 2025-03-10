from src.utils.base_schema import base_arquitecture
from langchain_core.prompts import ChatPromptTemplate

def enum_to_list(enum_class, category):
    return [
        {"id": member.name, "node_name": member.value, "node_category": category}
        for member in enum_class
        ]

def generate_prompt(node_focus, system_prompt, human_prompt):

    node_predecessor = [node['source'] for node in base_arquitecture['edges'] if node['target']==node_focus]
    node_predecessor_text = ', '.join(node_predecessor)

    completed_prompt = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("human", human_prompt)
        ]
    ).partial(
        node=node_focus,
        node_predecessor_text=node_predecessor_text)

    return completed_prompt

def get_edge_type(source_category, target_category, base_arquitecture):

    edge_type = [
        edge['edge_type']
        for edge in base_arquitecture['edges']
        if edge["source"] == source_category
        and edge["target"] == target_category
    ]

    return edge_type[0]

def clean_nodes(nodes, is_dataframe=False):

    if is_dataframe:
        nodes = nodes.to_dict(orient='records')

    cleaned_nodes = []
    for node in nodes:
        cleaned_node = {'_key' if k == 'id' else k: v for k, v in node.items()}
        if isinstance(cleaned_node.get('node_category'), list):
            cleaned_node['node_category'] = ', '.join(cleaned_node['node_category'])
        cleaned_nodes.append(cleaned_node)

    return cleaned_nodes

def clean_edges(edges, source, target):
    cleaned_edges = []
    
    for edge in edges:
        cleaned_edge = {
            '_from' if k == 'source' else '_to' if k == 'target' else k: v
            for k, v in edge.items()
        }
        
        if '_from' in cleaned_edge:
            cleaned_edge['_from'] = f"{source}/{cleaned_edge['_from']}"
        if '_to' in cleaned_edge:
            cleaned_edge['_to'] = f"{target}/{cleaned_edge['_to']}"
        
        cleaned_edges.append(cleaned_edge)

    return cleaned_edges