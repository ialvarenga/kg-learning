"""
Graph schema definitions for different knowledge graph extraction domains.
"""

from typing import Literal

SchemaType = Literal["book", "medical"]


def _get_book_schema() -> tuple[list[str], list[str]]:
    """Schema tailored for extracting knowledge graphs from novels and literary works."""
    node_labels = [
        # People & characters
        "Person",
        "Character",
        # Social structures
        "Family",
        "Organization",
        "Group",
        "SocialClass",
        # Places & settings
        "Place",
        "Location",
        "Building",
        # Narrative elements
        "Event",
        "Conflict",
        "Scene",
        # Objects & artifacts
        "Object",
        "Artifact",
        # Abstract / thematic
        "Concept",
        "Theme",
        "Belief",
        "Emotion",
        # Time
        "TimePeriod",
    ]
    rel_types = [
        # Kinship
        "PARENT_OF",
        "CHILD_OF",
        "SIBLING_OF",
        "MARRIED_TO",
        "RELATIVE_OF",
        # Professional / social hierarchy
        "WORKS_FOR",
        "EMPLOYS",
        "MENTORS",
        "COLLEAGUE_OF",
        "SERVANT_OF",
        # Interpersonal
        "FRIENDS_WITH",
        "ENEMIES_WITH",
        "LOVES",
        "BETRAYS",
        "RIVALS_WITH",
        # Narrative / plot
        "PARTICIPATES_IN",
        "WITNESSES",
        "CAUSES",
        "PART_OF",
        # Spatial / temporal
        "LOCATED_IN",
        "TRAVELS_TO",
        "BORN_IN",
        "DIED_IN",
        # Ownership / association
        "OWNS",
        "ASSOCIATED_WITH",
        "BELONGS_TO",
    ]
    return node_labels, rel_types


def _get_medical_schema() -> tuple[list[str], list[str]]:
    """Schema tailored for extracting knowledge graphs from medical and scientific texts."""
    basic_node_labels = ["Object", "Entity", "Group", "Person", "Organization", "Place"]
    academic_node_labels = ["ArticleOrPaper", "PublicationOrJournal"]
    medical_node_labels = [
        "Anatomy", "BiologicalProcess", "Cell", "CellularComponent",
        "CellType", "Condition", "Disease", "Drug",
        "EffectOrPhenotype", "Exposure", "GeneOrProtein", "Molecule",
        "MolecularFunction", "Pathway",
    ]
    node_labels = basic_node_labels + academic_node_labels + medical_node_labels
    rel_types = [
        "ACTIVATES", "AFFECTS", "ASSESSES", "ASSOCIATED_WITH",
        "AUTHORED", "BIOMARKER_FOR",
    ]
    return node_labels, rel_types


_SCHEMA_REGISTRY: dict[str, callable] = {
    "book": _get_book_schema,
    "medical": _get_medical_schema,
}


def get_graph_schema(schema_type: SchemaType = "medical") -> tuple[list[str], list[str]]:
    """
    Return node labels and relationship types for the given schema type.

    Args:
        schema_type: One of 'book' or 'medical'.

    Returns:
        A tuple of (node_labels, rel_types).

    Raises:
        ValueError: If an unknown schema_type is provided.
    """
    if schema_type not in _SCHEMA_REGISTRY:
        raise ValueError(
            f"Unknown schema type '{schema_type}'. "
            f"Available options: {list(_SCHEMA_REGISTRY.keys())}"
        )
    return _SCHEMA_REGISTRY[schema_type]()
