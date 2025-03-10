from src.ingestion.helpers import enum_to_list
from src.utils.enums import *

competence_nodes = enum_to_list(Competence, "competence")
industry_nodes = enum_to_list(Industry, "industry")
responsibility_level_nodes = enum_to_list(ResponsibilityLevel, "responsibility_level")
profession_nodes = enum_to_list(Profession, "profession")
experience_level_nodes = enum_to_list(ExperienceLevel, "experience_level")
specialization_area_nodes = enum_to_list(SpecializationArea, "specialization_area")
organizational_culture_nodes = enum_to_list(OrganizationalCulture, "organizational_culture")