"""Schema-driven typed extraction (Stage C).

This package is additive and parallel to the existing FactClaim → synthesis
pipeline. It introduces:

- ``schemas.py``  — Pydantic models and SCHEMA_REGISTRY for typed fields
- ``models.py``   — ``ExtractionResult``: the parallel typed artifact
- ``repair.py``   — ``SchemaRepairAttempt``: retry helper that asks the LLM to
                    fix a ValidationError-rejected response
- ``extractor.py``— ``SchemaExtractor``: orchestrates retrieve → prompt → parse
                    → validate → (repair) for one field

None of these touch ``FactClaim``, ``SynthesisResult``, ``PipelineRunResult``
(existing fields), or replay bundles. ``schema_validation=False`` by default so
no existing call site is affected.
"""
