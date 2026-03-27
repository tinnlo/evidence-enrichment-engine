from evidence_enrichment.context.resolver import ContextResolver


def test_context_resolver_loads_stage_entries_with_budget_metadata() -> None:
    resolver = ContextResolver(__import__("pathlib").Path("context/context_manifest.yaml"))
    bundle = resolver.resolve(entity_id="microsoft", field_name="hq_country")
    analysis_stage = bundle.stages["analysis"]
    assert bundle.task_name == "hq_country_resolution"
    assert analysis_stage.used_chars <= analysis_stage.budget_chars
    assert analysis_stage.entries
    assert any(entry.included for entry in analysis_stage.entries)
