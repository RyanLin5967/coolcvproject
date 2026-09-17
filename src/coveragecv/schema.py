"""Strict public coverage declarations. Missing labels never establish absence."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CoverageState = Literal["unknown", "positive_only", "exhaustive", "verified_absent"]
Split = Literal["train", "valid", "test"]
STATES = ("unknown", "positive_only", "exhaustive", "verified_absent")
NEGATIVE_ALLOWED = frozenset(("exhaustive", "verified_absent"))


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Source(Contract):
    id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    annotations: str
    images: str
    split: Split
    class_map: dict[str, str]
    coverage: dict[str, CoverageState] = Field(default_factory=dict)
    overrides: dict[str, dict[str, CoverageState]] = Field(default_factory=dict)
    evidence: str = Field(min_length=1)
    attribution: str = Field(min_length=1)


class CompileSpec(Contract):
    schema_version: Literal[1] = 1
    classes: list[str] = Field(min_length=1)
    sources: list[Source] = Field(min_length=1)

    @model_validator(mode="after")
    def references(self):
        if len(set(self.classes)) != len(self.classes) or any(not c.strip() for c in self.classes):
            raise ValueError("classes must be nonempty and unique; their order defines model indices")
        ids = [s.id for s in self.sources]
        if len(set(ids)) != len(ids):
            raise ValueError("source IDs must be unique")
        known = set(self.classes)
        for source in self.sources:
            keys = set(source.coverage) | set(source.class_map.values())
            for override in source.overrides.values():
                keys.update(override)
            if keys - known:
                raise ValueError(f"source {source.id}: unknown canonical classes {sorted(keys-known)}")
        return self


class DiagnosticError(ValueError):
    def __init__(self, code: str, message: str, **context):
        self.code, self.context = code, context
        super().__init__(message)

    def as_dict(self):
        return {"code": self.code, "message": str(self), "context": self.context}
