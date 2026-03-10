from __future__ import annotations

from shared.contracts.ingestion import CleaningStrategy

from inference.indexing.cleaning.base import DocumentCleaner
from inference.indexing.cleaning.strategies import BasicCleaner, DeepCleaner, MedicalGuidelineDeepCleaner, NoOpCleaner


class CleanerFactory:
    @staticmethod
    def create(strategy: CleaningStrategy) -> DocumentCleaner:
        if strategy == "none":
            return NoOpCleaner()
        if strategy == "basic":
            return BasicCleaner()
        if strategy == "deep":
            return DeepCleaner()
        if strategy == "medical_guideline_deep":
            return MedicalGuidelineDeepCleaner()
        raise ValueError(f"Unsupported cleaning strategy: {strategy}")
