from src.ingestion.helpers import generate_prompt
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Literal

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    model="gpt-4o-mini",
    temperature=0
)

# Classes

class Activity(BaseModel):

    name: str = Field(description="Name of the activity")
    weight: float = Field(description="Weight or relative importance to the competence that is aiming to improve")
    duration: int = Field(description="Estimated duration to accomplish the activity (in days)")

class NodeDetail(BaseModel):

    node_name: str = Field(
        description="Name of the node"
    )
    action_plan: List[Activity] = Field(
        description="Given a node name and previous conditions, add a list of activities."
    )

class NodeGenerator(BaseModel):
    node_type: Literal["skill", "knowledge", "attitude"] = Field(
        description="Type of node"
    )
    node_detail: List[NodeDetail] = Field(
        description="Given a competency and certain conditions, add a list of node details."
    )

# Prompt

base_system_prompt = """
    You are an expert in professional development and competency-based workforce planning.
    Your role is to generate structured development plans tailored to specific conditions.

    You will be asked to generate a structured plan of actionable activities based on a list of: {node}
    Each {node} if affected by conditions such as: {node_predecessor_text}.

    Instructions
    - node_type: Insert {node}.
    - node_detail: Create a list of exact 8 key {node} relevant to the provided competency, profession, and responsibility level.
      Each {node} represents a crucial area of expertise necessary for professional growth.

    Each {node} must contain:
    - node_name: The name of the {node}, directly linked to the given context. It should be precise and relevant.
    - action_plan: A structured set of activities (exact 3 per {node}) designed to develop the corresponding {node} effectively.

    Each activity must contain:
    - name: A clear and concise title describing the specific learning or training action.
    - weight: A numerical value between 0.0 and 1.0, indicating the relative importance of this activity in {node} development. Higher values mean greater relevance.
    - duration: An estimated time frame required to complete the activity. This should reflect realistic effort based on standard learning methodologies.

    Guidelines for Generation
    - Ensure that the selected {node} is logically derived from {node_predecessor_text}.
    - Activities should be realistic, actionable, and measurable, ensuring they contribute meaningfully to {node} acquisition.
    - Use industry-specific terminology where applicable, maintaining consistency with professional development standards.
    - Keep the response structured, concise, and free of additional explanations or formatting beyond what is required.

    Strictly adhere to the specified format and constraints. Do not include extra commentary, footnotes, or explanatory text.
    Only generate the structured content based on the given parameters.
"""

skill_human_prompt = """
    Context:
    Competence: {competence}
    Profession: {profession}
    Responsibility Level: {responsibility_level}
"""

knowledge_human_prompt = """
    Context:
    Competence: {competence}
    Specialization area: {specialization_area}
    Industry: {industry}
"""

attitude_human_prompt = """
    Context:
    Competence: {competence}
    Organizational culture: {organizational_culture}
    Experience level: {experience_level}
"""

structured_llm = llm.with_structured_output(NodeGenerator)

skill_prompt = generate_prompt('skill', base_system_prompt, skill_human_prompt)
skill_chain = skill_prompt | structured_llm

knowledge_prompt = generate_prompt('knowledge', base_system_prompt, knowledge_human_prompt)
knowledge_chain = knowledge_prompt | structured_llm

attitude_prompt = generate_prompt('attitude', base_system_prompt, attitude_human_prompt)
attitude_chain = attitude_prompt | structured_llm