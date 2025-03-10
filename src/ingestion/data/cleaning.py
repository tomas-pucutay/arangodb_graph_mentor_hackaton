from src.utils.base_schema import base_arquitecture
from src.ingestion.generators.nodes import *
from src.ingestion.helpers import clean_edges, clean_nodes

import cudf
import joblib
import pandas as pd

# Load data
skill_nodes = joblib.load('skill_nodes.pkl')
skill_edges = joblib.load('skill_edges.pkl')
knowledge_nodes = joblib.load('knowledge_nodes.pkl')
knowledge_edges = joblib.load('knowledge_edges.pkl')
attitude_nodes = joblib.load('attitude_nodes.pkl')
attitude_edges = joblib.load('attitude_edges.pkl')

total_nodes = skill_nodes + knowledge_nodes + attitude_nodes
total_edges = skill_edges + knowledge_edges + attitude_edges
nodes_df = pd.DataFrame(total_nodes)
edges_df = pd.DataFrame(total_edges)

# Data manipulation on nodes

# Driver nodes

driver_nodes = nodes_df[nodes_df['node_category'] != "action_plan"].copy()
driver_nodes = cudf.DataFrame(driver_nodes)
driver_nodes = driver_nodes.drop(columns=['duration']).rename(columns={'id': 'old_id'})

driver_category_list = (
    driver_nodes
    .sort_values(['node_name','node_category'], ascending=[True,True])
    .groupby('node_name')['node_category']
    .unique()
    .reset_index()
)

first_driver_name = (
    driver_nodes
    .groupby('node_name')['old_id']
    .first()
    .reset_index(name='id')
)

driver_nodes = (
    driver_nodes
    .drop(columns=['node_category'])
    .merge(
        driver_category_list,
        on='node_name',
        how='left'
    )
    .merge(
        first_driver_name,
        on='node_name',
        how='left'
    )
)

driver_nodes_id_rename = driver_nodes[['old_id','id']]

driver_nodes = (
    driver_nodes
    [['id', 'node_name', 'node_category']]
    .drop_duplicates()
)

# Data manipulation on nodes

# Action plan nodes

action_plan_nodes = nodes_df[nodes_df['node_category'] == "action_plan"].copy()
action_plan_nodes = cudf.DataFrame(action_plan_nodes)
action_plan_nodes = action_plan_nodes.rename(columns={'id': 'old_id'})

median_duration = (
    action_plan_nodes
    .groupby('node_name')['duration']
    .median()
    .reset_index(name='median_duration')
)

first_action_plan_name = (
    action_plan_nodes
    .groupby('node_name')['old_id']
    .first()
    .reset_index(name='id')
)

action_plan_nodes = (
    action_plan_nodes
    .merge(
        median_duration,
        on='node_name',
        how='left'
    )
    .merge(
        first_action_plan_name,
        on='node_name',
        how='left'
    )
)

action_plan_nodes_id_rename = action_plan_nodes[['old_id','id']]

action_plan_nodes = (
    action_plan_nodes
    [['id', 'node_name', 'node_category', 'median_duration']]
    .rename(columns={'median_duration':'duration'})
    .drop_duplicates(subset=['id', 'node_name'], keep='first')
)

action_plan_nodes['node_category'] = (
    action_plan_nodes['node_category']
    .to_pandas()
    .map(lambda x: [x])
)

# Data manipulation on edges

# Edges
edges_df = cudf.DataFrame(edges_df)

factor_to_driver = list(set([
    edge['edge_type']
    for edge in base_arquitecture['edges']
    if edge['target'] in ['skill','knowledge','attitude']
    ]))

driver_to_action_plan = list(set([
    edge['edge_type']
    for edge in base_arquitecture['edges']
    if edge['target'] in ['action_plan']
    ]))

action_plan_to_competence = list(set([
    edge['edge_type']
    for edge in base_arquitecture['edges']
    if edge['target'] in ['competence']
    ]))

# Factors to Drivers
factor_to_driver_edges = edges_df[edges_df['edge_type'].isin(factor_to_driver)]

factor_to_driver_edges = (
    factor_to_driver_edges
    .rename(columns={'target':'old_target'})
    .merge(driver_nodes_id_rename, left_on='old_target', right_on='old_id')
    .rename(columns={'id':'target'})
    [['source','target','edge_type']]
    .drop_duplicates()
)

# Drivers to Action Plan
driver_to_action_plan_edges = edges_df[edges_df['edge_type'].isin(driver_to_action_plan)]

driver_to_action_plan_edges = (
    driver_to_action_plan_edges
    .rename(columns={'source':'old_source', 'target':'old_target'})
    .merge(driver_nodes_id_rename, left_on='old_source', right_on='old_id')
    .rename(columns={'id':'source'})
    .merge(action_plan_nodes_id_rename, left_on='old_target', right_on='old_id')
    .rename(columns={'id':'target'})
    [['source','target','edge_type']]
    .drop_duplicates()
)

# Action Plan to Competence
action_plan_to_competence_edges = edges_df[edges_df['edge_type'].isin(action_plan_to_competence)]

action_plan_to_competence_edges = (
    action_plan_to_competence_edges
    .rename(columns={'source':'old_source'})
    .merge(action_plan_nodes_id_rename, left_on='old_source', right_on='old_id')
    .rename(columns={'id':'source'})
    [['source','target','edge_type', 'weight']]
    .groupby(['source','target','edge_type'])
    .agg(weight=('weight','mean'))
    .reset_index()
)

# Adjust edges
factor_to_driver_edges = factor_to_driver_edges.to_dict(orient='records')
driver_to_action_plan_edges = driver_to_action_plan_edges.to_dict(orient='records')
action_plan_to_competence_edges = action_plan_to_competence_edges.to_dict(orient='records')

competence_to_driver_edges = [edge for edge in factor_to_driver_edges if edge['source'].startswith('CMP')]
factor_to_driver_edges = [edge for edge in factor_to_driver_edges if not edge['source'].startswith('CMP')]

# Final nodes
competence_nodes = clean_nodes(competence_nodes)
factor_nodes = (
    clean_nodes(industry_nodes) + clean_nodes(responsibility_level_nodes) +
    clean_nodes(profession_nodes) + clean_nodes(experience_level_nodes) +
    clean_nodes(specialization_area_nodes) + clean_nodes(organizational_culture_nodes)
)
driver_nodes = clean_nodes(driver_nodes, is_dataframe=True)
action_plan_nodes = clean_nodes(action_plan_nodes, is_dataframe=True)

# Final edges
competence_to_driver_edges = clean_edges(competence_to_driver_edges, 'Competence', 'Driver')
factor_to_driver_edges = clean_edges(factor_to_driver_edges, 'Factor', 'Driver')
driver_to_action_plan_edges = clean_edges(driver_to_action_plan_edges, 'Driver', 'Action_Plan')
action_plan_to_competence_edges = clean_edges(action_plan_to_competence_edges, 'Action_Plan', 'Competence')