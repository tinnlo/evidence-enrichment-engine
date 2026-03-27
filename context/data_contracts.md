Contracts:

- SearchQueryPlan drives retrieval order and query variants.
- ParsedDocument is the unit passed into evidence assessment and analysis.
- FactClaim is the unit passed from analysis to synthesis.
- SynthesisResult is the only structure allowed to set the final output value.
- PipelineRunResult must retain context, trace, and artifact references for inspection.

