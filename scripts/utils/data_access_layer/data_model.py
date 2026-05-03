from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TemplateConfig:
    template_id: int
    layer_id: str
    question_template: str
    answer_type: str
    commands: List[str]
    benchmark_layer: str
    requires_rule: Optional[str] = None
    context: Optional[str] = None
    metrics: Optional[str] = None
    subtask: Optional[str] = None


@dataclass
class SceneRecord:
    scene_id: str
    file_path: str
    objects: List[str]
    features_map: Dict[str, List[str]]
    template_ids: List[int]
    fields: Dict[str, str]
    material_map: Dict[str, Dict[str, List[str]]]


@dataclass
class SceneUnderstandingRecord:
    scene_id: str
    file_path: str
    space_relations: Dict[str, object]
    object_relations: Dict[str, object]
    space_source_column: str
    spatial_source_column: str
    fields: Dict[str, str]
