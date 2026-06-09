from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from encoder.feature_spec import FeatureSpec
from preprocessing.schema import DatasetSchema


@dataclass
class DatasetMetadata:
    feature_spec: FeatureSpec
    schema: DatasetSchema
    num_samples: int = 0
    num_episodes: int = 0
    source_files: Sequence[str] | None = None

    def __post_init__(self) -> None:
        self.source_files = tuple(self.source_files or ())
